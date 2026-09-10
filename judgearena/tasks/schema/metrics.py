"""Declarative requests for battle-dataframe metrics."""

from __future__ import annotations

from pydantic import Field, model_validator

from judgearena.tasks.schema.base import StrictFrozenModel


class MetricSpec(StrictFrozenModel):
    """One named calculation over a battle dataframe."""

    metric: str = Field(min_length=1)
    breakdown_by: tuple[str, ...] = ()
    """Each field produces a separate breakdown; fields are not combined."""
    parameters: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_breakdown_by(self) -> MetricSpec:
        if any(not field for field in self.breakdown_by):
            raise ValueError("metric breakdown_by fields must not be empty")
        if len(set(self.breakdown_by)) != len(self.breakdown_by):
            raise ValueError("metric breakdown_by fields must not contain duplicates")
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
