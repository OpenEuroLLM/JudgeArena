"""Schema for judge meta-evaluation against human arena labels."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.pairwise import PairwiseJudgeSpec, ScoringSpec


class MetaEvalProtocol(StrictFrozenModel):
    """Policy for scoring a judge against human-labeled arena battles.

    Unlike the ELO protocol there is no model under evaluation and no
    generation step: both completions already exist in the arena, and the
    judge itself is the subject of the experiment.
    """

    runner: Literal["meta_eval"]
    arena: str = Field(min_length=1)
    judge: PairwiseJudgeSpec
    scoring: ScoringSpec
