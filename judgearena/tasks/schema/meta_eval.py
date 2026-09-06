"""Schema for judge meta-evaluation against human arena labels."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import NoBaseline
from judgearena.tasks.schema.metrics import ScoringSpec
from judgearena.tasks.schema.pairwise import PairwiseJudgeSpec

_META_EVAL_METRICS = {
    "meta_eval_agreement",
    "meta_eval_elo_gap",
    "meta_eval_ranking",
}


class MetaEvalProtocol(StrictFrozenModel):
    """Policy for scoring a judge against human-labeled arena battles."""

    runner: Literal["meta_eval"]
    arena: str = Field(min_length=1)
    baseline: NoBaseline
    judge: PairwiseJudgeSpec
    scoring: ScoringSpec

    @model_validator(mode="after")
    def _validate_scoring(self) -> MetaEvalProtocol:
        for request in self.scoring.metrics:
            if request.metric not in _META_EVAL_METRICS:
                raise ValueError(
                    f"unsupported meta-evaluation metric: {request.metric}"
                )
            if request.group_by:
                raise ValueError(
                    f"meta-evaluation metric {request.metric} does not support group_by"
                )
        return self
