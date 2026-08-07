"""Runtime scoring adapters for pairwise preference tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from judgearena.benchmarks.pairwise import scoring_alpaca_eval, scoring_arena_hard
from judgearena.utils.eval import PrefSummary, compute_pref_summary


@dataclass(frozen=True)
class ScoringResult:
    """Complete output produced by one scoring pass."""

    summary: PrefSummary
    metrics: dict[str, float | None] = field(default_factory=dict)
    grouped_results: dict[str, object] = field(default_factory=dict)
    scoring_details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PairwiseScorer:
    """Pairwise scoring implementation."""

    score: Callable[[pd.DataFrame], ScoringResult]
    check_requirements: Callable[[], None] | None = None


def _score_win_rate(battles: pd.DataFrame) -> ScoringResult:
    return ScoringResult(summary=compute_pref_summary(battles["pref"]))


PAIRWISE_SCORERS: dict[str, PairwiseScorer] = {
    "pairwise_win_rate": PairwiseScorer(score=_score_win_rate),
    "arena_hard_score": PairwiseScorer(score=scoring_arena_hard.score),
    "alpaca_eval_lc_winrate": PairwiseScorer(
        score=scoring_alpaca_eval.score,
        check_requirements=scoring_alpaca_eval.check_requirements,
    ),
}
