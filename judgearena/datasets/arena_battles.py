"""Dataset adapter for human preference battles used by ELO tasks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.arenas_utils import (
    KNOWN_ARENAS,
    load_arena_dataframe,
)
from judgearena.tasks.schema import (
    EloProtocol,
    HuggingFaceDatasetSource,
    ResolvedTaskSpec,
)


def _task_sources(
    task: ResolvedTaskSpec,
) -> tuple[EloProtocol, dict[str, HuggingFaceDatasetSource]]:
    protocol = task.spec.protocol
    if not isinstance(protocol, EloProtocol):
        raise ValueError(f"Task {task.task!r} does not define an ELO protocol.")
    if protocol.arena not in {*KNOWN_ARENAS, "LMArena"}:
        raise ValueError(f"Unsupported ELO arena {protocol.arena!r}.")

    sources: dict[str, HuggingFaceDatasetSource] = {}
    for source in task.spec.dataset.sources.values():
        if not isinstance(source, HuggingFaceDatasetSource):
            raise ValueError("ELO arena sources must be Hugging Face datasets.")
        if source.repo_id in sources:
            raise ValueError(f"Duplicate ELO arena source {source.repo_id!r}.")
        sources[source.repo_id] = source
    return protocol, sources


def download_task_sources(task: ResolvedTaskSpec, _local_dir: Path) -> None:
    """Download every pinned human-battle source declared by the task."""
    _, sources = _task_sources(task)
    for source in sources.values():
        snapshot_download(
            repo_id=source.repo_id,
            repo_type="dataset",
            revision=source.revision,
            allow_patterns=source.allow_patterns or None,
            force_download=False,
        )


def load_task_battles(task: ResolvedTaskSpec, _local_dir: Path) -> pd.DataFrame:
    """Load and normalize the task's pinned human preference battles."""
    protocol, sources = _task_sources(task)
    return load_arena_dataframe(protocol.arena, dataset_sources=sources)
