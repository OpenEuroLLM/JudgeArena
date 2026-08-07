"""Runtime scoring adapters for pairwise preference tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from judgearena.utils.eval import PrefSummary, compute_pref_summary


@dataclass(frozen=True)
class ScoringInputs:
    """Per-battle judging outcomes handed to a scorer.

    With ``swap_mode="both"`` the entries are ordered all-direct then
    all-reversed, and the completion lists follow the judged positions (so the
    reversed half has A and B swapped).
    """

    prefs: pd.Series
    completions_a: list[str] | None = None
    completions_b: list[str] | None = None


@dataclass(frozen=True)
class PairwiseScorer:
    """Scoring implementation and the semantics of its primary metric."""

    primary_metric: str
    higher_is_better: bool
    summarize: Callable[[ScoringInputs], PrefSummary]
    report_metadata: Callable[[ScoringInputs], dict[str, object]] | None = None
    """Optional scorer-specific details merged into the run report metadata."""


def _summarize_win_rate(inputs: ScoringInputs) -> PrefSummary:
    return compute_pref_summary(inputs.prefs)


PAIRWISE_SCORERS = {
    "pairwise_win_rate": PairwiseScorer(
        primary_metric="winrate",
        higher_is_better=True,
        summarize=_summarize_win_rate,
    ),
}
