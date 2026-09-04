"""Schema for judge meta-evaluation against human arena labels."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import NoBaseline
from judgearena.tasks.schema.metrics import ScoringSpec
from judgearena.tasks.schema.pairwise import PairwiseJudgeSpec


class MetaEvalProtocol(StrictFrozenModel):
    """Policy for scoring a judge against human-labeled arena battles."""

    runner: Literal["meta_eval"]
    arena: str = Field(min_length=1)
    baseline: NoBaseline
    judge: PairwiseJudgeSpec
    scoring: ScoringSpec
