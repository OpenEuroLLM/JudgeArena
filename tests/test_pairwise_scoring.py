"""Tests for runtime pairwise scoring adapters."""

import pandas as pd
import pytest

from judgearena.benchmarks.pairwise.scoring import PAIRWISE_SCORERS


def test_pairwise_win_rate_scorer_owns_metric_semantics():
    scorer = PAIRWISE_SCORERS["pairwise_win_rate"]

    summary = scorer.summarize(pd.Series([0.0, 1.0, 0.5]))

    assert scorer.primary_metric == "winrate"
    assert scorer.higher_is_better is True
    assert summary.num_battles == 3
    assert summary.winrate == pytest.approx(0.5)
