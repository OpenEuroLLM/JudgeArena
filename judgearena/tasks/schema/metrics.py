"""Declarative requests for battle-dataframe metrics."""

from __future__ import annotations

from pydantic import Field, model_validator

from judgearena.tasks.schema.base import StrictFrozenModel


class MetricSpec(StrictFrozenModel):
    """One named calculation over a battle dataframe."""

    metric: str = Field(min_length=1)
    group_by: tuple[str, ...] = ()
    parameters: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_group_by(self) -> MetricSpec:
        if any(not field for field in self.group_by):
            raise ValueError("metric group_by fields must not be empty")
        if len(set(self.group_by)) != len(self.group_by):
            raise ValueError("metric group_by fields must not contain duplicates")
        return self


class ScoringSpec(StrictFrozenModel):
    """Ordered metric calculations for one battle-producing protocol."""

    metrics: tuple[MetricSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_metrics(self) -> ScoringSpec:
        names = [item.metric for item in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("scoring metrics must not contain duplicate names")
        return self
