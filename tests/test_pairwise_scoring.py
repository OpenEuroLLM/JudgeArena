"""Tests for runtime pairwise scoring adapters."""

import pandas as pd
import pytest

from judgearena.benchmarks.pairwise.scoring import PAIRWISE_SCORERS
from judgearena.benchmarks.pairwise.scoring.alpaca_eval import (
    _official_annotations,
    _summarize,
)


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
    result = PAIRWISE_SCORERS["arena_hard_score"].score(
        _battles([0.0, 0.25, 0.5, 0.75, 1.0, None])
    )
    summary = result.summary

    assert summary.num_wins == 4
    assert summary.num_losses == 4
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
    assert first.scoring_details["confidence_level"] == 0.9
    assert first.scoring_details["confidence_quantiles"] == [0.05, 0.95]


def test_arena_hard_score_empty_prefs_yield_no_ci():
    result = PAIRWISE_SCORERS["arena_hard_score"].score(_battles([None, None]))

    assert result.metrics["score_ci_low"] is None
    assert result.metrics["score_ci_high"] is None


def test_arena_hard_score_drops_both_orders_when_one_is_unparseable():
    battles = _battles(
        [0.0, 0.25, None, 0.75],
        instruction_index=["q0", "q1", "q0", "q1"],
        orientation=["direct", "direct", "reversed", "reversed"],
    )

    summary = PAIRWISE_SCORERS["arena_hard_score"].score(battles).summary

    assert summary.num_wins == 1
    assert summary.num_losses == 1
    assert summary.num_ties == 0
    assert summary.num_missing == 2
    assert summary.winrate == pytest.approx(0.5)


def test_arena_hard_v2_reports_official_scores_per_category():
    battles = _battles(
        [0.0, 1.0, 0.0, 1.0],
        instruction_index=["hard", "creative", "hard", "creative"],
        orientation=["direct", "direct", "reversed", "reversed"],
        category=[
            "hard_prompt",
            "creative_writing",
            "hard_prompt",
            "creative_writing",
        ],
    )

    result = PAIRWISE_SCORERS["arena_hard_score"].score(battles)
    per_category = result.grouped_results["category"]

    assert per_category["hard_prompt"]["winrate"] == 1.0
    assert per_category["hard_prompt"]["baseline_model"] == "baseline-model"
    assert per_category["hard_prompt"]["score_ci_low"] == 1.0
    assert per_category["hard_prompt"]["score_ci_high"] == 1.0
    assert per_category["creative_writing"]["winrate"] == 0.0
    assert per_category["creative_writing"]["score_ci_low"] == 0.0
    assert per_category["creative_writing"]["score_ci_high"] == 0.0
    assert result.scoring_details["official_scope"] == "per_category"
    assert result.scoring_details["aggregate_score_is_official"] is False
    assert "score_ci_low" not in result.metrics
    assert "score_ci_high" not in result.metrics


def test_alpaca_eval_summary_is_mean_preference_over_parsed_battles():
    summary = _summarize(_battles([0.25, 0.75, None]))

    assert summary.winrate == pytest.approx(0.5)
    assert summary.num_battles == 3
    assert summary.num_missing == 1


def test_alpaca_eval_official_annotations_mapping():
    annotations = _official_annotations(_battles([0.25, None]))

    assert annotations["preference"].tolist() == [1.75, 0.0]
    assert annotations["index"].tolist() == [0, 1]
    assert str(annotations["index"].dtype).startswith("int")
    assert annotations["generator_2"].unique().tolist() == ["model-under-test"]
    assert annotations["generator_1"].unique().tolist() == ["baseline-model"]
    assert annotations["output_2"].tolist() == ["m", "mm"]
    assert annotations["output_1"].tolist() == ["b", "bbb"]


def test_alpaca_eval_lc_winrate_matches_pinned_reference_value():
    pytest.importorskip("alpaca_eval")
    import numpy as np

    scorer = PAIRWISE_SCORERS["alpaca_eval_lc_winrate"]
    rng = np.random.default_rng(0)
    n = 805
    battles = _battles(
        rng.uniform(0, 1, n).tolist(),
        completion_model=["x" * int(v) for v in rng.integers(50, 2000, n)],
        completion_baseline=["y" * int(v) for v in rng.integers(50, 2000, n)],
    )

    result = scorer.score(battles)

    if result.metrics["lc_winrate"] is None:
        pytest.skip("LC computation degraded (likely offline)")
    assert result.metrics["lc_winrate"] == pytest.approx(48.29068772368286, abs=0.5)
    assert result.metrics["raw_winrate"] == pytest.approx(48.20877535856965, abs=1e-6)
