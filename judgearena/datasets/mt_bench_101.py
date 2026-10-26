"""Dataset adapter for the YAML-defined MT-Bench-101 task."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from judgearena.datasets.data import CompletionData
from judgearena.datasets.registry import DatasetAdapter, register_dataset
from judgearena.datasets.registry import source_cache_dir as _task_cache_dir
from judgearena.tasks.schema import GitRawSource

MT_BENCH_101_TURN2_ONLY_TASKS = {"CM", "AR", "CR", "FR", "SC", "SA"}
MT_BENCH_101_REFERENCE_TASKS = {"MR", "GR"}
MT_BENCH_101_TASK_TO_ABILITY = {
    "CM": "memory",
    "AR": "understanding",
    "SI": "understanding",
    "TS": "interference",
    "CC": "interference",
    "CR": "rephrasing",
    "FR": "rephrasing",
    "SC": "reflection",
    "SA": "reflection",
    "MR": "reasoning",
    "GR": "reasoning",
    "IC": "questioning",
    "PI": "questioning",
}
MT_BENCH_101_TASK_TO_DOMAIN = {
    "CM": "perceptivity",
    "AR": "perceptivity",
    "SI": "perceptivity",
    "TS": "perceptivity",
    "CC": "perceptivity",
    "CR": "adaptability",
    "FR": "adaptability",
    "SC": "adaptability",
    "SA": "adaptability",
    "MR": "adaptability",
    "GR": "adaptability",
    "IC": "interactivity",
    "PI": "interactivity",
}


def _git_raw_url(source: GitRawSource) -> str:
    repository = source.repository.rstrip("/")
    github_prefix = "https://github.com/"
    if repository.startswith(github_prefix):
        project = repository.removeprefix(github_prefix)
        return (
            f"https://raw.githubusercontent.com/{project}/{source.revision}/"
            f"{source.path}"
        )
    return f"{repository}/raw/{source.revision}/{source.path}"


def expand_mt_bench_101_records(records: list[dict]) -> pd.DataFrame:
    """Expand dialogue JSONL records into golden-context turn rows."""
    rows: list[dict] = []
    for rec in records:
        task_name = rec.get("task")
        if task_name not in MT_BENCH_101_TASK_TO_ABILITY:
            raise ValueError(
                f"Unknown MT-Bench-101 task {task_name!r} in record: {rec}"
            )
        history = rec.get("history")
        if not isinstance(history, list):
            raise ValueError(
                "Invalid MT-Bench-101 record: expected list in field 'history', "
                f"got {type(history)}"
            )
        dialogue_id = rec.get("id")
        start_turn = 2 if task_name in MT_BENCH_101_TURN2_ONLY_TASKS else 1
        for turn_pos, turn in enumerate(history, start=1):
            if turn_pos < start_turn:
                continue
            if not isinstance(turn, dict):
                raise ValueError(
                    "Invalid MT-Bench-101 record: each turn in 'history' must be a dict."
                )
            user_message = str(turn.get("user") or "")
            reference_answer = str(turn.get("bot") or "")
            golden_context = [
                {
                    "user": str(prev_turn.get("user") or ""),
                    "bot": str(prev_turn.get("bot") or ""),
                }
                for prev_turn in history[: turn_pos - 1]
            ]
            rows.append(
                {
                    "instruction_index": len(rows),
                    "dialogue_id": dialogue_id,
                    "dialogue_uid": f"{task_name}:{dialogue_id}",
                    "task": task_name,
                    "ability": MT_BENCH_101_TASK_TO_ABILITY[task_name],
                    "domain": MT_BENCH_101_TASK_TO_DOMAIN[task_name],
                    "turn_index": turn_pos,
                    "golden_context": golden_context,
                    "user_message": user_message,
                    "reference_answer": reference_answer,
                    "requires_reference": task_name in MT_BENCH_101_REFERENCE_TASKS,
                    "instruction": user_message,
                }
            )
    return pd.DataFrame(rows)


@register_dataset("mt_bench_101")
class MTBench101Adapter(DatasetAdapter):
    def _benchmark_source(self) -> GitRawSource:
        source = self.task.spec.dataset.sources.get("benchmark")
        if not isinstance(source, GitRawSource):
            raise ValueError(
                f"Task {self.task.task!r} must define a git_raw source named 'benchmark'."
            )
        return source

    def _dataset_path(self) -> Path:
        return (
            _task_cache_dir(self.task, self.cache_dir)
            / Path(self._benchmark_source().path).name
        )

    def download(self) -> None:
        """Download the pinned MT-Bench-101 JSONL if it is missing."""
        if self.task.spec.dataset.adapter != "mt_bench_101":
            raise ValueError(
                f"Task {self.task.task!r} does not use the MT-Bench-101 adapter."
            )
        dataset_path = self._dataset_path()
        if dataset_path.exists():
            return
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        source = self._benchmark_source()
        try:
            urlretrieve(_git_raw_url(source), dataset_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download MT-Bench-101 from the pinned git_raw source. "
                f"If you are offline, place the file at {dataset_path}."
            ) from exc

    def load_data(self) -> CompletionData:
        return CompletionData.from_frames(
            self.load_instructions(),
        )

    def load_instructions(self) -> pd.DataFrame:
        """Load MT-Bench-101 and expand dialogues into turn-level evaluation items."""
        self.download()
        dataset_path = self._dataset_path()
        records: list[dict] = []
        with dataset_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return expand_mt_bench_101_records(records)
