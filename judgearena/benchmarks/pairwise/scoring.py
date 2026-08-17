"""Runtime scoring adapters for pairwise preference tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from judgearena.utils.eval import PrefSummary, compute_pref_summary


@dataclass(frozen=True)
class PairwiseScorer:
    """Scoring implementation and the semantics of its primary metric.

    Scorers receive one judged battle per row, canonically oriented as model
    versus baseline. Every runner guarantees a float64 ``pref`` column where
    0 is a model win, 0.5 is a tie, 1 is a baseline win, and NaN is
    unparseable. Individual scorers document any additional columns they need.
    """

    primary_metric: str
    higher_is_better: bool
    summarize: Callable[[pd.DataFrame], PrefSummary]
    report_metadata: Callable[[pd.DataFrame], dict[str, object]] | None = None
    """Optional scorer-specific details merged into the run report metadata."""
    check_requirements: Callable[[], None] | None = None
    """Optional pre-run check that raises when a scorer dependency is missing;
    runners call it before generation so a run fails fast."""


def _summarize_win_rate(battles: pd.DataFrame) -> PrefSummary:
    return compute_pref_summary(battles["pref"])


PAIRWISE_SCORERS = {
    "pairwise_win_rate": PairwiseScorer(
        primary_metric="winrate",
        higher_is_better=True,
        summarize=_summarize_win_rate,
    ),
}
