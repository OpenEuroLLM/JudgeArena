"""Shared scoring contracts for pairwise benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from judgearena.utils.eval import PrefSummary


@dataclass(frozen=True)
class ScoringResult:
    """Complete output produced by one scoring pass.

    Grouped results are keyed by dimensions such as ``category`` or ``turn``.
    """

    summary: PrefSummary
    metrics: dict[str, float | None] = field(default_factory=dict)
    grouped_results: dict[str, object] = field(default_factory=dict)
    scoring_details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PairwiseScorer:
    """Pairwise scoring implementation and its optional pre-run check."""

    score: Callable[[pd.DataFrame], ScoringResult]
    check_requirements: Callable[[], None] | None = None
