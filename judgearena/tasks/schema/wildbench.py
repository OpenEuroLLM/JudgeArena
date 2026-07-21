"""Schema for WildBench score and reward tasks."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import NoBaseline, OfficialOutputsBaseline
from judgearena.tasks.schema.pairwise import PairwiseJudgeSpec


class ConversationGeneration(StrictFrozenModel):
    """Continue a normalized multi-message conversation."""

    mode: Literal["conversation_chat"]


class WildBenchJudgeSpec(PairwiseJudgeSpec):
    """Official-prompt limits shared by WildBench judge modes."""

    max_words_to_evaluate: int = Field(default=1000, gt=0)


class WildBenchScoringSpec(StrictFrozenModel):
    """Versioned scorer and task-owned reward defaults."""

    adapter: str = Field(min_length=1)
    default_length_penalty_chars: int | None = Field(default=None, ge=0)


class WildBenchProtocol(StrictFrozenModel):
    """Policy used by the specialized WildBench V2 runner."""

    runner: Literal["wildbench"]
    mode: Literal["score", "reward"]
    generation: ConversationGeneration
    baseline: NoBaseline | OfficialOutputsBaseline = Field(discriminator="strategy")
    judge: WildBenchJudgeSpec
    scoring: WildBenchScoringSpec

    @model_validator(mode="after")
    def _validate_mode(self) -> WildBenchProtocol:
        if self.mode == "score":
            if not isinstance(self.baseline, NoBaseline):
                raise ValueError("WildBench score mode does not use a baseline")
            if self.scoring.default_length_penalty_chars is not None:
                raise ValueError(
                    "WildBench score mode does not support a length penalty"
                )
            return self

        if not isinstance(self.baseline, OfficialOutputsBaseline):
            raise ValueError("WildBench reward mode requires official baseline outputs")
        if not self.baseline.references:
            raise ValueError(
                "WildBench reward mode requires at least one baseline reference"
            )
        if len(set(self.baseline.references)) != len(self.baseline.references):
            raise ValueError(
                "WildBench baseline references must not contain duplicates"
            )
        return self
