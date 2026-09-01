"""Tests for runtime pairwise scoring adapters."""

import pandas as pd
import pytest

from judgearena.benchmarks.pairwise.scoring import (
    PAIRWISE_SCORERS,
    resolve_pairwise_scorer,
)
from judgearena.benchmarks.pairwise.scoring.alpaca_eval import (
    _official_annotations,
    _summarize,
)
from judgearena.benchmarks.pairwise.scoring.alpaca_eval import (
    score as score_alpaca_eval,
)
from judgearena.benchmarks.pairwise.scoring.arena_hard import _style_features


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


def test_pairwise_scorer_resolver_reports_available_names():
    assert (
        resolve_pairwise_scorer("pairwise_win_rate")
        is PAIRWISE_SCORERS["pairwise_win_rate"]
    )

    with pytest.raises(ValueError, match="Unknown pairwise scorer.*available"):
        resolve_pairwise_scorer("missing")


def test_pairwise_win_rate_scorer_owns_metric_semantics():
    result = PAIRWISE_SCORERS["pairwise_win_rate"].score(
        _battles([0.0, 0.0, 1.0, None])
    )

    assert result.summary.num_battles == 4
    assert result.summary.num_missing == 1
    assert result.summary.winrate == pytest.approx(2 / 3)


def test_arena_hard_score_weights_decisive_battles_three_to_one():
    result = PAIRWISE_SCORERS["arena_hard_v01_score"].score(
        _battles([0.0, 0.25, 0.75, None])
    )
    summary = result.summary

    assert (summary.num_wins, summary.num_losses, summary.num_ties) == (2, 1, 0)
    assert summary.num_missing == 1
    assert summary.num_battles == 4
    assert summary.winrate == pytest.approx(0.8)


def test_arena_hard_v01_matches_upstream_overall_bootstrap_fixture():
    scorer = PAIRWISE_SCORERS["arena_hard_v01_score"]
    battles = _battles(
        [0.0, 0.5, 0.75] * 10,
        category="arena-hard-v0.1",
    )

    first = scorer.score(battles)
    second = scorer.score(battles)

    assert first.summary.winrate == pytest.approx(0.7)
    assert (
        first.metrics
        == second.metrics
        == {
            "score_ci_low": pytest.approx(0.6),
            "score_ci_high": pytest.approx(0.8),
        }
    )
    assert first.grouped_results == {}
    assert first.scoring_details == {
        "decisive_weight": 3,
        "bootstrap_rounds": 100,
        "confidence_level": 0.95,
        "confidence_quantiles": [0.025, 0.975],
        "official_scope": "overall",
    }


def test_arena_hard_score_empty_prefs_yield_no_ci():
    result = PAIRWISE_SCORERS["arena_hard_v01_score"].score(_battles([None, None]))

    assert result.metrics["score_ci_low"] is None
    assert result.metrics["score_ci_high"] is None


def test_arena_hard_score_drops_both_orders_when_one_is_unparseable():
    battles = _battles(
        [0.0, 0.25, None, 0.75],
        instruction_index=["q0", "q1", "q0", "q1"],
        orientation=["direct", "direct", "reversed", "reversed"],
    )

    summary = PAIRWISE_SCORERS["arena_hard_v01_score"].score(battles).summary

    assert summary.num_wins == 1
    assert summary.num_losses == 1
    assert summary.num_ties == 0
    assert summary.num_battles == 4
    assert summary.num_missing == 2
    assert summary.winrate == pytest.approx(0.5)


def test_arena_hard_v2_reports_joint_official_scores_per_category(monkeypatch):
    from judgearena.benchmarks.pairwise.scoring import arena_hard

    monkeypatch.setattr(arena_hard, "BOOTSTRAP_ROUNDS", 3)
    hard_id = "0001b527ced3428d"
    battles = _battles(
        [0.0, 1.0, 0.0, 1.0],
        instruction_index=[hard_id, "creative", hard_id, "creative"],
        baseline=[
            "o3-mini-2025-01-31",
            "gemini-2.0-flash-001",
            "o3-mini-2025-01-31",
            "gemini-2.0-flash-001",
        ],
        judge="gpt-4.1",
        judge_prompt_preset="arena-hard",
        judge_temperature=0.0,
        judge_max_out_tokens=16000,
        orientation=["direct", "direct", "reversed", "reversed"],
        category=[
            "hard_prompt",
            "creative_writing",
            "hard_prompt",
            "creative_writing",
        ],
    )

    result = PAIRWISE_SCORERS["arena_hard_v20_score"].score(battles)
    per_category = result.grouped_results["category"]
    hard_prompt = per_category["hard_prompt"]

    assert hard_prompt["raw_winrate"] == 1.0
    assert hard_prompt["baseline_model"] == "o3-mini-2025-01-31"
    assert 0.0 <= hard_prompt["score_ci_low"] <= hard_prompt["winrate"]
    assert hard_prompt["winrate"] <= hard_prompt["score_ci_high"] <= 1.0
    assert hard_prompt["scoring_method"] == "joint_style_controlled_bt"
    assert hard_prompt["official_population_complete"] is False
    assert per_category["creative_writing"]["winrate"] == 0.0
    assert per_category["creative_writing"]["score_ci_low"] == 0.0
    assert per_category["creative_writing"]["score_ci_high"] == 0.0
    assert per_category["creative_writing"]["scoring_method"] == "weighted_mean"
    assert result.scoring_details["official_scope"] == "per_category"
    assert result.scoring_details["aggregate_score_is_official"] is False
    assert result.scoring_details["category_methods"] == {
        "creative_writing": "weighted_mean",
        "hard_prompt": "joint_style_controlled_bt",
    }
    assert "score_ci_low" not in result.metrics
    assert "score_ci_high" not in result.metrics


def test_arena_hard_v2_reproduces_published_full_population_score():
    from judgearena.benchmarks.pairwise.scoring import arena_hard

    population = arena_hard._load_style_calibration()
    live = population.loc[
        (population["judge"] == "gpt-4.1") & (population["model"] == "deepseek-r1")
    ].copy()
    live["judge_prompt_preset"] = "arena-hard"
    live["judge_temperature"] = 0.0
    live["judge_max_out_tokens"] = 16000

    calibration, complete = arena_hard._select_calibration(live)
    result = arena_hard.score_v20(live)
    hard_prompt = result.grouped_results["category"]["hard_prompt"]

    assert complete is True
    assert "deepseek-r1" not in {
        arena_hard._fit_model_id(model) for model in calibration["model"]
    }
    assert hard_prompt["winrate"] == pytest.approx(0.48, abs=0.01)
    assert hard_prompt["score_ci_low"] < 0.48 < hard_prompt["score_ci_high"]


def test_arena_hard_v2_rejects_unmatched_calibration_protocol():
    battles = _battles(
        [0.5, 0.5],
        instruction_index=["0001b527ced3428d"] * 2,
        baseline="o3-mini-2025-01-31",
        judge="gpt-4.1",
        judge_prompt_preset="custom-prompt",
        judge_temperature=0.0,
        judge_max_out_tokens=16000,
        orientation=["direct", "reversed"],
        category="hard_prompt",
    )

    with pytest.raises(ValueError, match="requires judge_prompt_preset='arena-hard'"):
        PAIRWISE_SCORERS["arena_hard_v20_score"].score(battles)


def test_arena_hard_v2_rejects_unsupported_judge_calibration():
    battles = _battles(
        [0.5, 0.5],
        instruction_index=["0001b527ced3428d"] * 2,
        baseline="o3-mini-2025-01-31",
        judge="other-judge",
        judge_prompt_preset="arena-hard",
        judge_temperature=0.0,
        judge_max_out_tokens=16000,
        orientation=["direct", "reversed"],
        category="hard_prompt",
    )

    with pytest.raises(ValueError, match="no calibration"):
        PAIRWISE_SCORERS["arena_hard_v20_score"].score(battles)


def test_arena_hard_style_features_match_upstream_extraction():
    completion = """# Header
1. ordered
- unordered
**bold** and __also bold__
```
# ignored
- ignored
**ignored**
```
"""

    assert _style_features(completion).tolist() == [31.0, 1.0, 2.0, 2.0]


def test_alpaca_eval_summary_reports_raw_model_win_rate():
    summary = _summarize(_battles([0.25, 0.5, None]))

    assert summary.winrate == pytest.approx(0.625)
    assert summary.num_battles == 3
    assert summary.num_missing == 1


def test_alpaca_eval_official_annotations_mapping():
    annotations = _official_annotations(_battles([0.25, None]))

    assert annotations["preference"].tolist() == [1.75]
    assert annotations["index"].tolist() == [0]
    assert str(annotations["index"].dtype).startswith("int")
    assert annotations["generator_2"].unique().tolist() == ["model-under-test"]
    assert annotations["generator_1"].unique().tolist() == ["baseline-model"]
    assert annotations["output_2"].tolist() == ["m"]
    assert annotations["output_1"].tolist() == ["b"]


def test_alpaca_eval_scorer_excludes_missing_rows_from_upstream(monkeypatch):
    metrics = pytest.importorskip("alpaca_eval.metrics")
    captured = {}

    def fake_get_length_controlled_winrate(annotations, **_kwargs):
        captured["annotations"] = annotations.copy()
        return {
            "length_controlled_winrate": 75.0,
            "lc_standard_error": 1.0,
            "win_rate": 75.0,
        }

    monkeypatch.setattr(
        metrics,
        "get_length_controlled_winrate",
        fake_get_length_controlled_winrate,
    )

    result = score_alpaca_eval(_battles([0.25, None]))

    assert len(captured["annotations"]) == 1
    assert result.summary.num_missing == 1
    assert result.metrics["lc_winrate"] == 75.0


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

    assert result.metrics["lc_winrate"] == pytest.approx(48.29068772368286, abs=0.5)
    assert result.metrics["raw_winrate"] == pytest.approx(48.20877535856965, abs=1e-6)
