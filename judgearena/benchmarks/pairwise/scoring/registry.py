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
    "arena_hard_v01_score": PairwiseScorer(score=arena_hard.score_v01),
    "arena_hard_v20_score": PairwiseScorer(
        score=arena_hard.score_v20,
        check_runtime=arena_hard.check_v20_runtime,
    ),
    "alpaca_eval_lc_winrate": PairwiseScorer(score=alpaca_eval.score),
}


def resolve_pairwise_scorer(name: str) -> PairwiseScorer:
    """Return the scorer selected by a pairwise task."""
    try:
        return PAIRWISE_SCORERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown pairwise scorer {name!r}; available: {sorted(PAIRWISE_SCORERS)}"
        ) from exc
