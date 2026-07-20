"""Dataset-native baselines for pairwise generate-and-evaluate benchmarks."""

from __future__ import annotations

from collections.abc import Mapping

from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import CategoryDefaultsBaseline, TaskDefaultBaseline


def native_pairwise_baseline(task: str) -> str | Mapping[str, str] | None:
    """Return the baseline declared by a registered task."""
    resolved = get_packaged_task(task)
    if resolved is not None:
        baseline = resolved.spec.protocol.baseline
        if isinstance(baseline, TaskDefaultBaseline):
            return baseline.reference_id
        if isinstance(baseline, CategoryDefaultsBaseline):
            return baseline.references
        return None

    return None
