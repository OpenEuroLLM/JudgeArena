"""Runtime scoring adapters for pairwise preference tasks."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from judgearena.utils.eval import PrefSummary, compute_pref_summary

PairwiseScoreFn = Callable[[pd.DataFrame], PrefSummary]


def _score_win_rate(battles: pd.DataFrame) -> PrefSummary:
    """Summarize canonical pairwise preferences."""
    return compute_pref_summary(battles["pref"])


PAIRWISE_SCORERS: dict[str, PairwiseScoreFn] = {
    "pairwise_win_rate": _score_win_rate,
}
