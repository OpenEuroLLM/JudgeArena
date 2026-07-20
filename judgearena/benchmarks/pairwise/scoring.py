"""Runtime scoring adapters for pairwise preference tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from judgearena.utils.eval import PrefSummary, compute_pref_summary


@dataclass(frozen=True)
class PairwiseScorer:
    """Scoring implementation and the semantics of its primary metric."""

    name: str
    primary_metric: str
    higher_is_better: bool
    summarize: Callable[[pd.Series], PrefSummary]


_PAIRWISE_SCORERS = {
    "pairwise_win_rate": PairwiseScorer(
        name="pairwise_win_rate",
        primary_metric="winrate",
        higher_is_better=True,
        summarize=compute_pref_summary,
    )
}

PAIRWISE_SCORER_NAMES = frozenset(_PAIRWISE_SCORERS)

# Used by legacy tasks that have no YAML definition to declare a scorer.
DEFAULT_PAIRWISE_SCORER = "pairwise_win_rate"


def resolve_pairwise_scorer(name: str) -> PairwiseScorer:
    """Return the registered pairwise scorer named by a task definition."""
    try:
        return _PAIRWISE_SCORERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown pairwise scorer {name!r}; available: "
            f"{sorted(PAIRWISE_SCORER_NAMES)}"
        ) from exc
