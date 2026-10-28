"""Schema for the shared text and conversation pairwise evaluation protocol."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import BaselineSpec
from judgearena.tasks.schema.metrics import ScoringSpec


class SingleTurnGeneration(StrictFrozenModel):
    """Generate one answer from instruction text or adapter-supplied messages.

    provided_messages uses each row's role/content history without appending
    generated answers to other rows. instruction_id optionally sets its cache ID.
    """

    mode: Literal["single_turn_chat", "base_completion", "provided_messages"]
    default_temperature: float | None = Field(default=None, ge=0)
    default_max_out_tokens: int | None = Field(default=None, gt=0)
    default_truncate_input: bool = True
    """Whether generation uses the default input cap when none is specified."""


class MultiTurnGeneration(StrictFrozenModel):
    mode: Literal["multi_turn_chat"]
    system_prompt: str | None = None
    fallback_temperature: float = Field(default=0.0, ge=0, allow_inf_nan=False)
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


SwapMode = Literal["fixed", "both", "random"]


class JudgeSpec(StrictFrozenModel):
    """Prompt selection and model defaults shared by judging protocols."""

    prompt_options: dict[str, Any] = Field(default_factory=dict)
    use_default_engine_kwargs: bool = True
    default_prompt_preset: str = Field(min_length=1)
    default_temperature: float | None = None
    default_max_out_tokens: int | None = Field(default=None, gt=0)
    default_top_logprobs: int | None = Field(default=None, gt=0)
    category_prompts: dict[str, str] = Field(default_factory=dict)
    """Per-category prompt presets overriding ``default_prompt_preset`` (e.g. the
    Arena-Hard v2.0 creative-writing judge prompt)."""


class PairwiseJudgeSpec(JudgeSpec):
    score_temperature: float = Field(default=0.3, gt=0, allow_inf_nan=False)
    """Multiplier on independent score differences when converting them to preferences."""
    minimum_score_by: tuple[str, ...] = ()
    """Group candidate scores by these metadata columns before comparing minima."""
    turns: Literal["all", "last"] | tuple[Annotated[int, Field(gt=0)], ...] | None = (
        None
    )
    """Select conversation prefixes; None judges each prepared example intact."""
    default_swap_mode: SwapMode = "fixed"


class PairwiseProtocol(StrictFrozenModel):
    """Task-owned generation, baseline, judging, and scoring behavior."""

    runner: Literal["pairwise"]
    limit_by: str | None = Field(default=None, min_length=1)
    """Count complete groups in this column when applying n_instructions."""
    generation: SingleTurnGeneration | MultiTurnGeneration
    baseline: BaselineSpec
    judge: PairwiseJudgeSpec
    scoring: ScoringSpec
