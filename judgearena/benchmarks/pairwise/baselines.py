"""Dataset-native baselines for pairwise generate-and-evaluate benchmarks."""

from __future__ import annotations

from collections.abc import Mapping

from judgearena.datasets.m_arenahard import (
    M_ARENA_HARD_BASELINES,
    split_m_arena_hard_dataset,
)
from judgearena.datasets.mt_bench import MT_BENCH_BASELINES
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import CategoryDefaultsBaseline, TaskDefaultBaseline

LEGACY_PAIRWISE_BASELINES: dict[str, str | Mapping[str, str]] = {
    **M_ARENA_HARD_BASELINES,
    **MT_BENCH_BASELINES,
}


def native_pairwise_baseline(task: str) -> str | Mapping[str, str] | None:
    """Return the task-defined baseline, with fallback for unmigrated tasks."""
    resolved = get_packaged_task(task)
    if resolved is not None:
        baseline = resolved.spec.protocol.baseline
        if isinstance(baseline, TaskDefaultBaseline):
            return baseline.reference_id
        if isinstance(baseline, CategoryDefaultsBaseline):
            return baseline.references
        return None

    if task in LEGACY_PAIRWISE_BASELINES:
        return LEGACY_PAIRWISE_BASELINES[task]
    parsed_m_arena_hard = split_m_arena_hard_dataset(task)
    if parsed_m_arena_hard is not None:
        version_key, _lang_or_subset = parsed_m_arena_hard
        return LEGACY_PAIRWISE_BASELINES[version_key]
    return None
