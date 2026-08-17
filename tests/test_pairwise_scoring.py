"""Tests for runtime pairwise scoring adapters."""

import pandas as pd
import pytest

from judgearena.benchmarks.pairwise.scoring import PAIRWISE_SCORERS


def _battles(prefs: list, **overrides) -> pd.DataFrame:
    n = len(prefs)
    columns = {
        "instruction_index": [f"{i:04d}" for i in range(n)],
        "model": "model-under-test",
        "baseline": "baseline-model",
        "completion_model": ["m" * (i + 1) for i in range(n)],
        "completion_baseline": ["b" * (2 * i + 1) for i in range(n)],
        "pref": pd.Series(prefs, dtype="float64"),
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


def test_pairwise_win_rate_scorer_owns_metric_semantics():
    scorer = PAIRWISE_SCORERS["pairwise_win_rate"]

    result = scorer.score(_battles([0.0, 1.0, 0.5]))

    assert scorer.primary_metric == "winrate"
    assert scorer.higher_is_better is True
    assert result.summary.num_battles == 3
    assert result.summary.winrate == pytest.approx(0.5)
    assert result.metrics == {}
    assert result.breakdowns == {}
    assert result.methodology == {}
