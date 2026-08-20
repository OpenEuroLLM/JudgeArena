"""Dataset adapter for JudgeArena's packaged instruction/output tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.tasks.schema import HuggingFaceDatasetSource, ResolvedTaskSpec


def download_task_sources(task: ResolvedTaskSpec, local_dir: Path) -> None:
    """Download every Hugging Face source declared by a table-backed task."""
    if task.spec.dataset.adapter != "judgearena_tables":
        raise ValueError(
            f"Task {task.task!r} uses dataset adapter "
            f"{task.spec.dataset.adapter!r}, not 'judgearena_tables'."
        )
    local_dir.mkdir(exist_ok=True, parents=True)
    for name, source in task.spec.dataset.sources.items():
        if not isinstance(source, HuggingFaceDatasetSource):
            raise ValueError(
                f"Dataset source {name!r} for task {task.task!r} is not supported "
                "by the 'judgearena_tables' adapter."
            )
        snapshot_download(
            repo_id=source.repo_id,
            repo_type="dataset",
            revision=source.revision,
            allow_patterns=list(source.allow_patterns) or None,
            local_dir=local_dir,
            force_download=False,
        )


def load_task_instructions(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame:
    """Load a task's table and map its declared fields to runner names."""
    download_task_sources(task, local_tables_path)
    path = local_tables_path / "instructions" / f"{task.task}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Instruction table not found at {path}")
    df = pd.read_csv(path)

    fields = task.spec.dataset.fields
    field_mapping = {
        fields.id: "instruction_index",
        fields.instruction: "instruction",
    }
    if fields.category is not None:
        field_mapping[fields.category] = "category"
    missing = sorted(set(field_mapping) - set(df.columns))
    if missing:
        raise ValueError(
            f"Task {task.task!r} is missing declared dataset fields: {missing}."
        )
    return df.rename(columns=field_mapping)


def load_task_model_outputs(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame | None:
    """Load optional pre-generated model outputs for a table-backed task."""
    download_task_sources(task, local_tables_path)
    path = local_tables_path / "model_outputs" / f"{task.task}.csv.zip"
    if not path.exists():
        return None
    return pd.read_csv(path)
