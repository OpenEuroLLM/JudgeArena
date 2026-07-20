"""Dataset adapter for multilingual base-model fluency tasks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.tasks.schema import HuggingFaceDatasetSource, ResolvedTaskSpec


def _source(task: ResolvedTaskSpec) -> HuggingFaceDatasetSource:
    source = task.spec.dataset.sources.get("examples")
    if not isinstance(source, HuggingFaceDatasetSource):
        raise ValueError(
            f"Task {task.task!r} must define a Hugging Face dataset source "
            "named 'examples'."
        )
    if len(source.allow_patterns) != 1:
        raise ValueError(
            f"Task {task.task!r} must select exactly one fluency context file."
        )
    return source


def _contexts_dir(local_tables_path: Path) -> Path:
    """Reuse the historical data-root/contexts download location."""
    return local_tables_path.parent / "contexts"


def download_task_sources(task: ResolvedTaskSpec, local_tables_path: Path) -> None:
    """Download the task's pinned language context file."""
    if task.spec.dataset.adapter != "fluency":
        raise ValueError(f"Task {task.task!r} does not use the fluency adapter.")
    source = _source(task)
    snapshot_download(
        repo_id=source.repo_id,
        repo_type="dataset",
        revision=source.revision,
        allow_patterns=list(source.allow_patterns),
        local_dir=_contexts_dir(local_tables_path),
        force_download=False,
    )


def load_task_instructions(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame:
    """Load contexts and assign stable row-based instruction IDs."""
    download_task_sources(task, local_tables_path)
    source = _source(task)
    path = _contexts_dir(local_tables_path) / source.allow_patterns[0]
    frame = pd.read_csv(path)
    instruction_field = task.spec.dataset.fields.instruction
    if instruction_field not in frame.columns:
        raise ValueError(
            f"Task {task.task!r} is missing declared instruction field "
            f"{instruction_field!r}."
        )
    return pd.DataFrame(
        {
            "instruction_index": frame.index,
            "instruction": frame[instruction_field],
        }
    )


def load_task_model_outputs(task: ResolvedTaskSpec, local_tables_path: Path) -> None:
    """Fluency contexts do not ship pre-generated model completions."""
    return None
