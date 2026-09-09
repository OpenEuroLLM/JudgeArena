"""Dataset adapter for the YAML-defined MT-Bench-101 task."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from judgearena.tasks.schema import GitRawSource, ResolvedTaskSpec

MT_BENCH_101_TURN2_ONLY_TASKS = {"CM", "AR", "CR", "FR", "SC", "SA"}
MT_BENCH_101_REFERENCE_TASKS = {"MR", "GR"}
MT_BENCH_101_TASK_TO_ABILITY = {
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


def _benchmark_source(task: ResolvedTaskSpec) -> GitRawSource:
    source = task.spec.dataset.sources.get("benchmark")
    if not isinstance(source, GitRawSource):
        raise ValueError(
            f"Task {task.task!r} must define a git_raw source named 'benchmark'."
        )
    return source


def _task_cache_dir(task: ResolvedTaskSpec, local_tables_path: Path) -> Path:
    return local_tables_path / "_sources" / task.definition_task


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


def _dataset_path(task: ResolvedTaskSpec, local_tables_path: Path) -> Path:
    return (
        _task_cache_dir(task, local_tables_path)
        / Path(_benchmark_source(task).path).name
    )


def download_task_sources(task: ResolvedTaskSpec, local_tables_path: Path) -> None:
    """Download the pinned MT-Bench-101 JSONL if it is missing."""
    if task.spec.dataset.adapter != "mt_bench_101":
        raise ValueError(f"Task {task.task!r} does not use the MT-Bench-101 adapter.")
    dataset_path = _dataset_path(task, local_tables_path)
    if dataset_path.exists():
        return
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    source = _benchmark_source(task)
    try:
        urlretrieve(_git_raw_url(source), dataset_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download MT-Bench-101 from the pinned git_raw source. "
            f"If you are offline, place the file at {dataset_path}."
        ) from exc


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
                    "turn_index": turn_pos,
                    "golden_context": golden_context,
                    "user_message": user_message,
                    "reference_answer": reference_answer,
                    "requires_reference": task_name in MT_BENCH_101_REFERENCE_TASKS,
                    "instruction": user_message,
                }
            )
    return pd.DataFrame(rows)


def load_task_instructions(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame:
    """Load MT-Bench-101 and expand dialogues into turn-level evaluation items."""
    download_task_sources(task, local_tables_path)
    dataset_path = _dataset_path(task, local_tables_path)
    records: list[dict] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return expand_mt_bench_101_records(records)


def load_task_model_outputs(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame | None:
    """MT-Bench-101 has no packaged model outputs."""
    return None
