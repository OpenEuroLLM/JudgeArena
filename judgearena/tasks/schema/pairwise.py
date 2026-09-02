"""Schema for the common single-turn pairwise evaluation protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import BaselineSpec


class SingleTurnGeneration(StrictFrozenModel):
    # "single_turn_chat" prompts an instruction-tuned model through its chat
    # template; "base_completion" lets a base model continue the raw text.
    mode: Literal["single_turn_chat", "base_completion"]


SwapMode = Literal["fixed", "both", "random"]


class PairwiseJudgeSpec(StrictFrozenModel):
    default_prompt_preset: str = Field(min_length=1)
    default_swap_mode: SwapMode = "fixed"
    default_temperature: float | None = None
    default_max_out_tokens: int | None = Field(default=None, gt=0)
    default_top_logprobs: int | None = Field(default=None, gt=0)
    category_prompts: dict[str, str] = Field(default_factory=dict)
    """Per-category prompt presets overriding ``default_prompt_preset`` (e.g. the
    Arena-Hard v2.0 creative-writing judge prompt)."""


class MetricSpec(StrictFrozenModel):
    """One metric calculation over pairwise battles."""

    metric: str = Field(min_length=1)
    group_by: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_group_by(self) -> MetricSpec:
        if any(not field for field in self.group_by):
            raise ValueError("metric group_by fields must not be empty")
        if len(set(self.group_by)) != len(self.group_by):
            raise ValueError("metric group_by fields must not contain duplicates")
        return self


class ScoringSpec(StrictFrozenModel):
    metrics: tuple[MetricSpec, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_metrics(self) -> ScoringSpec:
        names = [item.metric for item in self.metrics]
        if len(set(names)) != len(names):
            raise ValueError("scoring metrics must not contain duplicate names")
        return self


class PairwiseProtocol(StrictFrozenModel):
    """Task-owned generation, baseline, judging, and scoring behavior."""

    runner: Literal["pairwise"]
    generation: SingleTurnGeneration
    baseline: BaselineSpec
    judge: PairwiseJudgeSpec
    scoring: ScoringSpec
