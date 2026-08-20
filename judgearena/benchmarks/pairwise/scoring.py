"""Runtime scoring adapters for pairwise preference tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from judgearena.utils.eval import PrefSummary, compute_pref_summary


@dataclass(frozen=True)
class PairwiseScorer:
    """Scoring implementation and the semantics of its primary metric."""

    primary_metric: str
    higher_is_better: bool
    summarize: Callable[[pd.Series], PrefSummary]


PAIRWISE_SCORERS = {
    "pairwise_win_rate": PairwiseScorer(
        primary_metric="winrate",
        higher_is_better=True,
        summarize=compute_pref_summary,
    )
}
