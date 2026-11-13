"""Legacy Elo input schemas, translated to pairwise settings by RunConfig."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import NoBaseline
from judgearena.tasks.schema.metrics import ScoringSpec
from judgearena.tasks.schema.pairwise import (
    MultiTurnGeneration,
    PairwiseJudgeSpec,
    SingleTurnGeneration,
)


class EloScoringSpec(ScoringSpec):
    """Task-owned defaults for fitting arena-anchored ratings."""

    default_soft: bool = True
    default_temperature: float = Field(default=0.3, gt=0)


class EloProtocol(StrictFrozenModel):
    """Accepted old task format; it has no separate execution runner."""

    runner: Literal["elo"]
    arena: str = Field(min_length=1)
    generation: SingleTurnGeneration | MultiTurnGeneration
    baseline: NoBaseline
    judge: PairwiseJudgeSpec
    scoring: EloScoringSpec
