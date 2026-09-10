"""Tests for runtime pairwise scoring adapters."""

import pandas as pd
import pytest

from judgearena.benchmarks.pairwise.scoring import PAIRWISE_SCORERS


def test_pairwise_win_rate_scorer_owns_metric_semantics():
    battles = pd.DataFrame({"pref": pd.Series([0.0, 0.0, 1.0, None], dtype=float)})

    summary = PAIRWISE_SCORERS["pairwise_win_rate"](battles)

    assert summary.num_battles == 4
    assert summary.num_missing == 1
    assert summary.winrate == pytest.approx(2 / 3)
