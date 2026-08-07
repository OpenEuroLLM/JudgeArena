"""AlpacaEval dataset adapter: instructions plus shipped baseline outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.tasks.schema import HuggingFaceDatasetSource, ResolvedTaskSpec

# The AlpacaEval 2.0 evaluation set: 805 instructions, each with the official
# GPT-4-Turbo baseline output inline (field "generator" names the model).
_BASELINE_FILE = "alpaca_eval_gpt4_baseline.json"


def _source(task: ResolvedTaskSpec) -> HuggingFaceDatasetSource:
    source = task.spec.dataset.sources.get("examples")
    if not isinstance(source, HuggingFaceDatasetSource):
        raise ValueError(
            f"Task {task.task!r} must define an 'examples' Hugging Face source."
        )
    return source


def download_task_sources(task: ResolvedTaskSpec, local_tables_path: Path) -> None:
    """Populate canonical instruction and output tables for one task."""
    if task.spec.dataset.adapter != "alpaca_eval":
        raise ValueError(f"Task {task.task!r} does not use the AlpacaEval adapter.")
    instructions_path = local_tables_path / "instructions" / f"{task.task}.csv"
    model_outputs_path = local_tables_path / "model_outputs" / f"{task.task}.csv.zip"
    if instructions_path.exists() and model_outputs_path.exists():
        return

    source = _source(task)
    snapshot_root = snapshot_download(
        repo_id=source.repo_id,
        repo_type="dataset",
        allow_patterns=list(source.allow_patterns) or None,
        force_download=False,
        revision=source.revision,
    )
    raw_df = pd.read_json(Path(snapshot_root) / _BASELINE_FILE)
    instructions_path.parent.mkdir(parents=True, exist_ok=True)
    model_outputs_path.parent.mkdir(parents=True, exist_ok=True)
    _build_instructions(raw_df).to_csv(instructions_path, index=False)
    _build_model_outputs(raw_df).to_csv(model_outputs_path, index=False)


def load_task_instructions(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame:
    """Load normalized instructions for a registered AlpacaEval task."""
    download_task_sources(task, local_tables_path)
    return pd.read_csv(local_tables_path / "instructions" / f"{task.task}.csv")


def load_task_model_outputs(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame | None:
    """Load the shipped baseline outputs for a registered AlpacaEval task."""
    download_task_sources(task, local_tables_path)
    return pd.read_csv(local_tables_path / "model_outputs" / f"{task.task}.csv.zip")


def _instruction_index(raw_df: pd.DataFrame) -> pd.Series:
    # Zero-padded positions keep lexicographic index order == file order.
    return pd.Series(
        [f"{i:04d}" for i in range(len(raw_df))], index=raw_df.index, dtype=str
    )


def _build_instructions(raw_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instruction_index": _instruction_index(raw_df),
            "instruction": raw_df["instruction"].astype(str),
        }
    )


def _build_model_outputs(raw_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instruction_index": _instruction_index(raw_df),
            "model": raw_df["generator"].astype(str),
            "output": raw_df["output"].astype(str),
        }
    )
