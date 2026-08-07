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
    result = PAIRWISE_SCORERS["pairwise_win_rate"].score(_battles([0.0, 1.0, 0.5]))

    assert result.summary.num_battles == 3
    assert result.summary.winrate == pytest.approx(0.5)


def test_arena_hard_score_weights_decisive_battles_three_to_one():
    # Graded prefs: [[A>>B]], [[A>B]], [[A=B]], [[B>A]], [[B>>A]], unparsed.
    # With one model vs one baseline, the official Bradley-Terry fit reduces
    # to the weighted win fraction, so winrate IS the official score.
    result = PAIRWISE_SCORERS["arena_hard_score"].score(
        _battles([0.0, 0.25, 0.5, 0.75, 1.0, None])
    )
    summary = result.summary

    assert summary.num_wins == 4  # 3x for A>>B + 1x for A>B
    assert summary.num_losses == 4  # 3x for B>>A + 1x for B>A
    assert summary.num_ties == 1
    assert summary.num_missing == 1
    assert summary.num_battles == 10
    assert summary.winrate == pytest.approx((4 + 0.5) / 9)


def test_arena_hard_score_bootstrap_ci_is_ordered_and_deterministic():
    scorer = PAIRWISE_SCORERS["arena_hard_score"]
    battles = _battles([0.0, 0.25, 0.5, 0.75, 1.0] * 4)

    first = scorer.score(battles)
    second = scorer.score(battles)

    assert first == second
    assert 0.0 <= first.metrics["score_ci_low"] <= first.metrics["score_ci_high"] <= 1.0
    assert first.scoring_details["decisive_weight"] == 3
    assert first.scoring_details["bootstrap_rounds"] == 100


def test_arena_hard_score_empty_prefs_yield_no_ci():
    result = PAIRWISE_SCORERS["arena_hard_score"].score(_battles([None, None]))

    assert result.metrics["score_ci_low"] is None
    assert result.metrics["score_ci_high"] is None
