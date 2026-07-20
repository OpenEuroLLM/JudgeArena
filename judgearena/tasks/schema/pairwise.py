"""Schema for the common single-turn pairwise evaluation protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import BaselineSpec


class SingleTurnGeneration(StrictFrozenModel):
    mode: Literal["single_turn_chat"]


SwapMode = Literal["fixed", "both"]


class PairwiseJudgeSpec(StrictFrozenModel):
    default_prompt: str = Field(min_length=1)
    parser: str = Field(min_length=1)
    default_swap_mode: SwapMode = "fixed"
    allowed_swap_modes: tuple[SwapMode, ...] = ("fixed", "both")
    default_temperature: float | None = None

    @model_validator(mode="after")
    def _default_must_be_allowed(self) -> PairwiseJudgeSpec:
        if not self.allowed_swap_modes:
            raise ValueError("allowed_swap_modes must not be empty")
        if self.default_swap_mode not in self.allowed_swap_modes:
            raise ValueError("default_swap_mode must be present in allowed_swap_modes")
        if len(set(self.allowed_swap_modes)) != len(self.allowed_swap_modes):
            raise ValueError("allowed_swap_modes must not contain duplicates")
        return self


class ScoringSpec(StrictFrozenModel):
    adapter: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    higher_is_better: bool


class PairwiseProtocol(StrictFrozenModel):
    """Task-owned generation, baseline, judging, and scoring behavior."""

    runner: Literal["pairwise"]
    generation: SingleTurnGeneration
    baseline: BaselineSpec
    judge: PairwiseJudgeSpec
    scoring: ScoringSpec
