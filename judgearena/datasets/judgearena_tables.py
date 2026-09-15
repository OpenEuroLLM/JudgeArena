"""Dataset adapter for JudgeArena's packaged instruction/output tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.tasks.schema import HuggingFaceDatasetSource, ResolvedTaskSpec


def _table_path(task: ResolvedTaskSpec, directory: str, suffix: str) -> Path:
    """Return an exact task-declared table path, falling back to its task ID."""
    paths = {
        Path(pattern)
        for source in task.spec.dataset.sources.values()
        if isinstance(source, HuggingFaceDatasetSource)
        for pattern in source.allow_patterns
        if pattern.startswith(f"{directory}/")
        and not any(character in pattern for character in "*?[")
    }
    if len(paths) > 1:
        raise ValueError(
            f"Task {task.task!r} declares multiple {directory!r} table paths."
        )
    return next(iter(paths), Path(directory) / f"{task.task}{suffix}")


def _instruction_path(task: ResolvedTaskSpec, local_tables_path: Path) -> Path:
    return local_tables_path / _table_path(task, "instructions", ".csv")


def _model_output_path(task: ResolvedTaskSpec, local_tables_path: Path) -> Path:
    return local_tables_path / _table_path(task, "model_outputs", ".csv.zip")


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


def _read_instruction_table(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame:
    path = _instruction_path(task, local_tables_path)
    if not path.exists():
        raise FileNotFoundError(f"Instruction table not found at {path}")
    frame = pd.read_csv(path)
    fields = task.spec.dataset.fields
    required = {fields.id, fields.instruction}
    if fields.category is not None:
        required.add(fields.category)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Task {task.task!r} is missing declared dataset fields: {missing}."
        )
    if frame[fields.id].duplicated().any():
        raise ValueError(f"Task {task.task!r} contains duplicate instruction IDs.")
    return frame


def load_task_instructions(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame:
    """Load a task's table and map its declared fields to runner names."""
    download_task_sources(task, local_tables_path)
    frame = _read_instruction_table(task, local_tables_path)
    fields = task.spec.dataset.fields
    if fields.id_strategy == "position":
        frame[fields.id] = range(len(frame))
    field_mapping = {
        fields.id: "instruction_index",
        fields.instruction: "instruction",
    }
    if fields.category is not None:
        field_mapping[fields.category] = "category"
    return frame.rename(columns=field_mapping)


def load_task_model_outputs(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame | None:
    """Load optional pre-generated model outputs for a table-backed task."""
    download_task_sources(task, local_tables_path)
    path = _model_output_path(task, local_tables_path)
    if not path.exists():
        return None
    outputs = pd.read_csv(path)
    if task.spec.dataset.fields.id_strategy != "position":
        return outputs

    instructions = _read_instruction_table(task, local_tables_path)
    source_id = task.spec.dataset.fields.id
    id_map = pd.Series(
        range(len(instructions)),
        index=instructions[source_id].astype(str),
        dtype=int,
    )
    mapped = outputs["instruction_index"].astype(str).map(id_map)
    if mapped.isna().any():
        unknown = outputs.loc[mapped.isna(), "instruction_index"].unique()[:5]
        raise ValueError(
            f"Task {task.task!r} model outputs contain unknown instruction IDs: "
            f"{unknown.tolist()}."
        )
    outputs["instruction_index"] = mapped.astype(int)
    return outputs
