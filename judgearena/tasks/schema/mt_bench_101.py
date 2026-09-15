"""Schema for the MT-Bench-101 golden-context single-answer protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import BaselineSpec
from judgearena.tasks.schema.metrics import ScoringSpec
from judgearena.tasks.schema.pairwise import PairwiseJudgeSpec


class GoldenContextGeneration(StrictFrozenModel):
    mode: Literal["golden_context_chat"]
    default_max_out_tokens: int | None = Field(default=None, gt=0)


class MTBench101JudgeSpec(PairwiseJudgeSpec):
    default_temperature: float = Field(default=0.6, ge=0)


class MTBench101Protocol(StrictFrozenModel):
    """Task policy for per-turn 1-10 grading with min-per-dialogue aggregation."""

    runner: Literal["mt_bench_101"]
    generation: GoldenContextGeneration
    baseline: BaselineSpec
    judge: MTBench101JudgeSpec
    scoring: ScoringSpec
