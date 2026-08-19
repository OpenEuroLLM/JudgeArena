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
    allowed_swap_modes: tuple[SwapMode, ...] = ("fixed", "both")
    default_temperature: float | None = None
    default_max_out_tokens: int | None = Field(default=None, gt=0)
    default_top_logprobs: int | None = Field(default=None, gt=0)
    category_prompts: dict[str, str] = Field(default_factory=dict)
    """Per-category prompt presets overriding ``default_prompt_preset`` (e.g. the
    Arena-Hard v2.0 creative-writing judge prompt)."""

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


class PairwiseProtocol(StrictFrozenModel):
    """Task-owned generation, baseline, judging, and scoring behavior."""

    runner: Literal["pairwise"]
    generation: SingleTurnGeneration
    baseline: BaselineSpec
    judge: PairwiseJudgeSpec
    scoring: ScoringSpec
