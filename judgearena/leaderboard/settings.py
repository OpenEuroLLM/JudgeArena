"""Release settings kept separate from the existing evaluation configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SamplingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    opponents: list[str] | None = Field(default=None, min_length=1)
    validation_fraction: float = Field(default=0.1, ge=0, lt=1)
    minimum_model_comparisons: int = Field(default=50, gt=0)
    comparisons_per_opponent: int = Field(default=10, gt=0)
    max_reference_comparisons: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_panel(self):
        if self.opponents is None:
            return self
        if len(set(self.opponents)) != len(self.opponents) or not all(self.opponents):
            raise ValueError("Opponents must be nonempty, unique model names")
        if (
            self.max_reference_comparisons is not None
            and self.max_reference_comparisons
            < self.comparisons_per_opponent * len(self.opponents)
        ):
            raise ValueError("Reference limit must accommodate the opponent panel")
        return self


class LeaderboardSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    labels: Literal["human", "judge"] = "human"
    sampling: SamplingSettings = Field(default_factory=SamplingSettings)


def configured_metrics(cfg):
    """Validate the ordinary metric requests used by a release."""
    from judgearena.benchmarks.elo.scoring import (
        BradleyTerryMetric,
        configure_frozen_metrics,
    )
    from judgearena.benchmarks.scoring import build_metrics

    configured = build_metrics(cfg.resolved_task.spec.protocol.scoring.metrics)
    for _, metric in configured:
        if isinstance(metric, Exception):
            raise metric
    if not configured:
        raise ValueError("A leaderboard requires at least one metric")
    ratings = []
    observed = []
    for request, metric in configure_frozen_metrics(cfg, configured):
        if isinstance(metric, BradleyTerryMetric):
            ratings.append((request.name or request.metric, metric))
        else:
            observed.append((request, metric))
    return ratings, observed
