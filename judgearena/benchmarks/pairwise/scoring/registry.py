"""Built-in pairwise scorer registry."""

from __future__ import annotations

import pandas as pd

from judgearena.benchmarks.pairwise.scoring import alpaca_eval, arena_hard
from judgearena.benchmarks.pairwise.scoring.models import (
    PairwiseScorer,
    ScoringResult,
)
from judgearena.utils.eval import compute_pref_summary


def _score_win_rate(battles: pd.DataFrame) -> ScoringResult:
    return ScoringResult(summary=compute_pref_summary(battles["pref"]))


PAIRWISE_SCORERS: dict[str, PairwiseScorer] = {
    "pairwise_win_rate": PairwiseScorer(score=_score_win_rate),
    "arena_hard_score": PairwiseScorer(score=arena_hard.score),
    "alpaca_eval_lc_winrate": PairwiseScorer(
        score=alpaca_eval.score,
        check_requirements=alpaca_eval.check_requirements,
    ),
}
