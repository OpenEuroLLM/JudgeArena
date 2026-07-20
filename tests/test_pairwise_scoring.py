"""Tests for runtime pairwise scoring adapters."""

import pandas as pd
import pytest

from judgearena.benchmarks.pairwise.scoring import resolve_pairwise_scorer


def test_pairwise_win_rate_scorer_owns_metric_semantics():
    scorer = resolve_pairwise_scorer("pairwise_win_rate")

    summary = scorer.summarize(pd.Series([0.0, 1.0, 0.5]))

    assert scorer.primary_metric == "winrate"
    assert scorer.higher_is_better is True
    assert summary.num_battles == 3
    assert summary.winrate == pytest.approx(0.5)


def test_resolve_pairwise_scorer_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown pairwise scorer"):
        resolve_pairwise_scorer("missing")
