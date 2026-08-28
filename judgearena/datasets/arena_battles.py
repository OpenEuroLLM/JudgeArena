"""Dataset adapter for human preference battles used by arena-backed tasks."""

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
    MetaEvalProtocol,
    ResolvedTaskSpec,
)

ArenaProtocol = EloProtocol | MetaEvalProtocol
CANONICAL_WINNERS = frozenset({"model_a", "model_b", "tie"})
_WINNER_ALIASES_BY_ARENA = {
    "LMArena-100k": {"tie (bothbad)": "tie"},
    "LMArena-140k": {"both_bad": "tie"},
    "LMArena": {"tie (bothbad)": "tie", "both_bad": "tie"},
}


def _canonicalize_winners(battles: pd.DataFrame, *, arena: str) -> pd.DataFrame:
    """Map one arena's native verdicts to the shared battle-label contract."""
    normalized = battles.copy()
    normalized["winner"] = normalized["winner"].replace(
        _WINNER_ALIASES_BY_ARENA.get(arena, {})
    )
    invalid_mask = normalized["winner"].isna() | ~normalized["winner"].isin(
        CANONICAL_WINNERS
    )
    if invalid_mask.any():
        invalid = sorted(set(normalized.loc[invalid_mask, "winner"].astype(str)))
        raise ValueError(f"Unsupported winner labels for {arena}: {invalid}.")
    return normalized


def _task_sources(
    task: ResolvedTaskSpec,
) -> tuple[ArenaProtocol, dict[str, HuggingFaceDatasetSource]]:
    protocol = task.spec.protocol
    if not isinstance(protocol, ArenaProtocol):
        raise ValueError(f"Task {task.task!r} does not define an arena protocol.")
    if protocol.arena not in {*KNOWN_ARENAS, "LMArena"}:
        raise ValueError(f"Unsupported arena {protocol.arena!r}.")

    sources: dict[str, HuggingFaceDatasetSource] = {}
    for source in task.spec.dataset.sources.values():
        if not isinstance(source, HuggingFaceDatasetSource):
            raise ValueError("Arena sources must be Hugging Face datasets.")
        if source.repo_id in sources:
            raise ValueError(f"Duplicate arena source {source.repo_id!r}.")
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
    battles = load_arena_dataframe(protocol.arena, dataset_sources=sources)
    return _canonicalize_winners(battles, arena=protocol.arena)
