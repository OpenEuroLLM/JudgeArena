"""Runtime scoring adapters for pairwise preference tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from judgearena.utils.eval import PrefSummary, compute_pref_summary


@dataclass(frozen=True)
class ScoringInputs:
    """One judged battle per row, canonically oriented (model vs baseline).

    Columns: ``instruction_index``, ``model``, ``baseline``,
    ``completion_model``, ``completion_baseline``, and ``pref`` (float64,
    P(baseline wins) in [0, 1], NaN = unparseable). Rows follow judged order:
    direct battles first, then the reversed duplicates under
    ``swap_mode="both"`` (same canonical orientation, so instruction_index
    values repeat there).
    """

    battles: pd.DataFrame


@dataclass(frozen=True)
class PairwiseScorer:
    """Scoring implementation and the semantics of its primary metric."""

    primary_metric: str
    higher_is_better: bool
    summarize: Callable[[ScoringInputs], PrefSummary]
    report_metadata: Callable[[ScoringInputs], dict[str, object]] | None = None
    """Optional scorer-specific details merged into the run report metadata."""
    check_available: Callable[[], None] | None = None
    """Optional pre-run check that raises when a scorer dependency is missing;
    runners call it before generation so a run fails fast."""


def _summarize_win_rate(inputs: ScoringInputs) -> PrefSummary:
    return compute_pref_summary(inputs.battles["pref"])


PAIRWISE_SCORERS = {
    "pairwise_win_rate": PairwiseScorer(
        primary_metric="winrate",
        higher_is_better=True,
        summarize=_summarize_win_rate,
    ),
}
