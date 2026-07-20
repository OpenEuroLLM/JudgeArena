"""Dataset-native baselines for pairwise generate-and-evaluate benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import (
    CategoryDefaultsBaseline,
    ResolvedTaskSpec,
    TaskDefaultBaseline,
)


@dataclass(frozen=True)
class BaselinePlan:
    """Row-aligned baseline assignment for model B."""

    baseline_by_index: pd.Series

    @classmethod
    def flat(cls, model: str, *, index: pd.Index) -> BaselinePlan:
        return cls(
            baseline_by_index=pd.Series(model, index=index, name="model_B", dtype=str)
        )

    @classmethod
    def per_row(cls, series: pd.Series) -> BaselinePlan:
        return cls(baseline_by_index=series.astype(str).rename("model_B"))

    @property
    def unique_models(self) -> list[str]:
        return sorted(self.baseline_by_index.dropna().unique().tolist())

    @property
    def is_flat(self) -> bool:
        return len(self.unique_models) == 1

    @property
    def single_model(self) -> str:
        if not self.is_flat:
            raise ValueError(
                "BaselinePlan is per-row; use baseline_by_index for row-level lookups."
            )
        return self.unique_models[0]

    @property
    def display_name(self) -> str:
        return self.single_model if self.is_flat else "+".join(self.unique_models)

    def aligned_to(self, index: pd.Index) -> pd.Series:
        return self.baseline_by_index.loc[index]


def resolve_baseline_plan(
    *,
    task_id: str,
    task: ResolvedTaskSpec | None,
    runtime_baseline: str | None,
    instructions: pd.DataFrame,
) -> BaselinePlan:
    """Resolve a runtime override or the baseline declared by a task."""
    if runtime_baseline is not None:
        return BaselinePlan.flat(runtime_baseline, index=instructions.index)

    if task is None:
        raise ValueError(
            f"model.baseline is required for task {task_id!r}; no task baseline "
            "is registered."
        )

    baseline = task.spec.protocol.baseline
    if isinstance(baseline, TaskDefaultBaseline):
        return BaselinePlan.flat(baseline.reference_id, index=instructions.index)

    if isinstance(baseline, CategoryDefaultsBaseline):
        if "category" not in instructions.columns:
            raise ValueError(
                f"{task_id} requires a 'category' column for per-category "
                "baseline routing; re-run dataset download to regenerate the "
                "instructions table."
            )
        per_row = instructions["category"].map(baseline.references)
        if per_row.isna().any():
            unknown = sorted(
                instructions.loc[per_row.isna(), "category"].unique().tolist()
            )
            raise ValueError(
                f"Unknown baseline categories for {task_id}: {unknown}. "
                f"Known: {sorted(baseline.references)}"
            )
        return BaselinePlan.per_row(per_row)

    raise ValueError(
        f"model.baseline is required for task {task_id!r}; its "
        f"{baseline.strategy!r} baseline policy does not provide a model."
    )


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
