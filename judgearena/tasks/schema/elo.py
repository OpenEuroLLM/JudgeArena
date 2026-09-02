"""Schema for arena-anchored ELO rating tasks."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import NoBaseline
from judgearena.tasks.schema.metrics import ScoringSpec
from judgearena.tasks.schema.pairwise import PairwiseJudgeSpec, SingleTurnGeneration


class EloScoringSpec(ScoringSpec):
    """Task-owned defaults for fitting arena-anchored ratings."""

    default_soft: bool = True
    default_temperature: float = Field(default=0.3, gt=0)


class EloProtocol(StrictFrozenModel):
    """Policy used by the specialized arena battle and ELO runner."""

    runner: Literal["elo"]
    arena: str = Field(min_length=1)
    generation: SingleTurnGeneration
    baseline: NoBaseline
    judge: PairwiseJudgeSpec
    scoring: EloScoringSpec
