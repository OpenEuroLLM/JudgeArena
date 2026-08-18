"""Schema for official WildBench V2 tasks."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import NoBaseline


class ConversationGeneration(StrictFrozenModel):
    """Generate the final assistant turn from a complete conversation."""

    mode: Literal["conversation_chat"]
    default_temperature: float = 0.0
    default_top_p: float = Field(default=1.0, gt=0, le=1)
    default_max_out_tokens: int = Field(default=4096, gt=0)
    default_input_char_limit: int | None = Field(default=None, gt=0)


class WildBenchJudgeSpec(StrictFrozenModel):
    """Official judge prompt and runtime defaults."""

    prompt_file: str = Field(min_length=1)
    parser: str = Field(min_length=1)
    reference_judge: str = Field(min_length=1)
    max_words_to_eval: int | None = Field(default=None, gt=0)
    default_swap_mode: Literal["fixed"] = "fixed"
    allowed_swap_modes: tuple[Literal["fixed"], ...] = ("fixed",)
    default_temperature: float = 0.0
    default_max_out_tokens: int = Field(default=1024, gt=0)
    default_top_logprobs: int | None = None


class WildBenchScoringSpec(StrictFrozenModel):
    adapter: str = Field(min_length=1)


class WildBenchProtocol(StrictFrozenModel):
    """Official WB-Score execution policy."""

    runner: Literal["wildbench"]
    mode: Literal["score"]
    generation: ConversationGeneration
    baseline: NoBaseline
    judge: WildBenchJudgeSpec
    scoring: WildBenchScoringSpec
