"""Schema for judge meta-evaluation against reference labels."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import NoBaseline
from judgearena.tasks.schema.metrics import ScoringSpec
from judgearena.tasks.schema.pairwise import PairwiseJudgeSpec


class MetaEvalProtocol(StrictFrozenModel):
    """Score a judge on existing answer pairs with reference labels."""

    runner: Literal["meta_eval"]
    arena: str | None = Field(default=None, min_length=1)
    sampling: Literal["per_model", "uniform_pairs"] = "uniform_pairs"
    baseline: NoBaseline = NoBaseline(strategy="none")
    judge: PairwiseJudgeSpec
    scoring: ScoringSpec

    @model_validator(mode="before")
    @classmethod
    def _legacy_sampling(cls, values):
        # Legacy arena definitions implied per-model sampling.
        if isinstance(values, dict) and (
            values.get("sampling") == "arena_per_model"
            or ("sampling" not in values and values.get("arena") is not None)
        ):
            return {**values, "sampling": "per_model"}
        return values
