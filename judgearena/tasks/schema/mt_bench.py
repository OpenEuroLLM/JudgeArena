"""Schema owned by the specialized multi-turn MT-Bench protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import BaselineSpec
from judgearena.tasks.schema.metrics import ScoringSpec
from judgearena.tasks.schema.pairwise import PairwiseJudgeSpec


class MultiTurnGeneration(StrictFrozenModel):
    mode: Literal["multi_turn_chat"]
    category_temperatures: dict[str, float] = Field(default_factory=dict)
    default_max_out_tokens: int | None = Field(default=None, gt=0)
    default_seed: int | None = None

    @model_validator(mode="after")
    def _validate_temperatures(self) -> MultiTurnGeneration:
        invalid = {
            category: temperature
            for category, temperature in self.category_temperatures.items()
            if not category or temperature < 0
        }
        if invalid:
            raise ValueError(
                f"category temperatures require names and non-negative values: {invalid}"
            )
        return self


class MTBenchJudgeSpec(PairwiseJudgeSpec):
    turns_mode: Literal["both", "single", "multi"] = "both"
    fastchat_prompt_preset: str = Field(min_length=1)
    fastchat_temperature: float = Field(ge=0)
    reference_categories: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_reference_categories(self) -> MTBenchJudgeSpec:
        if any(not category for category in self.reference_categories):
            raise ValueError("reference categories must not be empty")
        if len(set(self.reference_categories)) != len(self.reference_categories):
            raise ValueError("reference categories must not contain duplicates")
        return self


class MTBenchProtocol(StrictFrozenModel):
    """Task policy used by the specialized multi-turn MT-Bench runner."""

    runner: Literal["mt_bench"]
    generation: MultiTurnGeneration
    baseline: BaselineSpec
    judge: MTBenchJudgeSpec
    scoring: ScoringSpec
