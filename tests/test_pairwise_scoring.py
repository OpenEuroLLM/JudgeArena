"""Behavior tests for composable pairwise metrics."""

import math

import numpy as np
import pandas as pd
import pytest

from judgearena.benchmarks.pairwise.scoring import (
    LengthControlledWinrateMetric,
    PairwiseWinRateMetric,
    collapse_pairwise_battles,
)
from judgearena.benchmarks.scoring import (
    available_metrics,
    build_metric,
    build_metrics,
    calculate_metrics,
    render_metrics,
)
from judgearena.tasks.schema import MetricSpec


def _calculate_metrics(
    battles: pd.DataFrame, requests: tuple[MetricSpec, ...]
) -> dict[str, dict[str, object]]:
    return calculate_metrics(battles, build_metrics(requests))


def _battle_rows(
    length_differences: np.ndarray,
    outcomes: np.ndarray,
) -> pd.DataFrame:
    baseline_length = 200
    return pd.DataFrame(
        {
            "instruction_index": range(len(outcomes)),
            "model": "candidate",
            "baseline": "reference",
            "completion_model": [
                "m" * (baseline_length + int(delta)) for delta in length_differences
            ],
            "completion_baseline": ["b" * baseline_length] * len(outcomes),
            "pref": 1 - outcomes,
            "orientation": "single",
        }
    )


def test_pairwise_win_rate_reports_candidate_results():
    battles = pd.DataFrame({"pref": [0.0, 0.25, 1.0, None]})

    result = PairwiseWinRateMetric().calculate(battles)

    assert result == {
        "num_battles": 4,
        "winrate": pytest.approx(2 / 3),
        "num_wins": 2,
        "num_losses": 1,
        "num_ties": 0,
        "num_missing": 1,
    }


def test_length_controlled_winrate_predicts_at_equal_length(monkeypatch):
    import judgearena.benchmarks.pairwise.scoring.metrics as metrics

    monkeypatch.setattr(metrics, "BOOTSTRAP_ROUNDS", 100)
    differences = np.arange(-100, 101, 10, dtype=float)
    scale = differences.std(ddof=1)
    expected = 0.4
    intercept = math.log(expected / (1 - expected))
    outcomes = 1 / (1 + np.exp(-(intercept + 0.8 * differences / scale)))

    result = LengthControlledWinrateMetric().calculate(
        _battle_rows(differences, outcomes)
    )

    assert result["winrate"] == pytest.approx(expected, abs=1e-5)
    assert result["confidence_interval"] is not None
    assert float(outcomes.mean()) != pytest.approx(expected)
    assert "raw_winrate" not in result


def test_length_control_requires_complete_answer_order_pairs():
    direct = _battle_rows(np.array([-10, 0, 10]), np.array([0.2, 0.5, 0.8]))
    direct["orientation"] = "direct"
    reversed_rows = direct.copy()
    reversed_rows["orientation"] = "reversed"
    reversed_rows.loc[1, "pref"] = np.nan
    battles = pd.concat([direct, reversed_rows], ignore_index=True)

    collapsed = collapse_pairwise_battles(battles)

    assert collapsed["n_parsed"].tolist() == [2, 1, 2]
    result = LengthControlledWinrateMetric().calculate(battles)
    assert result == {"num_pairs": 3, "num_scored": 2, "winrate": None}


def test_metrics_can_be_calculated_by_group():
    battles = pd.DataFrame(
        {
            "pref": [0.0, 1.0, 0.0, 0.0],
            "category": ["a", "a", "b", "b"],
        }
    )

    results = _calculate_metrics(
        battles,
        (MetricSpec(metric="pairwise_win_rate", group_by=("category",)),),
    )

    metric = results["pairwise_win_rate"]
    assert metric["winrate"] == 0.75
    assert [item["group"] for item in metric["groups"]["category"]] == ["a", "b"]
    assert [item["values"]["winrate"] for item in metric["groups"]["category"]] == [
        0.5,
        1.0,
    ]


def test_pairwise_metrics_reject_invalid_preferences():
    with pytest.raises(ValueError, match="finite values"):
        PairwiseWinRateMetric().calculate(pd.DataFrame({"pref": [0.0, float("inf")]}))


def test_bootstrap_does_not_replace_undefined_draws(monkeypatch):
    import judgearena.benchmarks.pairwise.scoring.metrics as metrics

    monkeypatch.setattr(metrics, "BOOTSTRAP_ROUNDS", 100)
    differences = np.array([-10, 0, 10], dtype=float)
    outcomes = np.array([0.0, 1.0, 0.0])

    result = LengthControlledWinrateMetric().calculate(
        _battle_rows(differences, outcomes)
    )

    assert result["winrate"] is not None
    assert result["confidence_interval"] is None


def test_grouped_metric_rejects_missing_column():
    with pytest.raises(ValueError, match="missing column 'category'"):
        _calculate_metrics(
            pd.DataFrame({"pref": [0.0, 1.0]}),
            (MetricSpec(metric="pairwise_win_rate", group_by=("category",)),),
        )


def test_collapse_rejects_duplicate_or_incomplete_orientations():
    direct = _battle_rows(np.array([0]), np.array([0.5]))
    direct["orientation"] = "direct"
    duplicate = pd.concat([direct, direct], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate orientations"):
        collapse_pairwise_battles(duplicate)

    reversed_row = direct.copy()
    reversed_row["orientation"] = "reversed"
    second_direct = _battle_rows(np.array([1]), np.array([0.5]))
    second_direct["instruction_index"] = 1
    second_direct["orientation"] = "direct"
    incomplete = pd.concat([direct, reversed_row, second_direct], ignore_index=True)
    with pytest.raises(ValueError, match="expected orientations"):
        collapse_pairwise_battles(incomplete)


def test_collapse_rejects_different_completions_across_orders():
    direct = _battle_rows(np.array([0]), np.array([0.5]))
    direct["orientation"] = "direct"
    reversed_row = direct.copy()
    reversed_row["orientation"] = "reversed"
    reversed_row["completion_model"] = "different"

    with pytest.raises(ValueError, match="different completions"):
        collapse_pairwise_battles(pd.concat([direct, reversed_row]))


def test_length_controlled_winrate_rejects_mixed_baselines():
    battles = _battle_rows(np.array([-10, 0, 10]), np.array([0.2, 0.5, 0.8]))
    battles.loc[2, "baseline"] = "other-reference"

    with pytest.raises(ValueError, match="exactly one baseline model"):
        LengthControlledWinrateMetric().calculate(battles)


def test_grouped_metrics_preserve_distinct_group_values():
    battles = pd.DataFrame(
        {"pref": [0.0, 0.0, 0.0, 0.0], "group": [1, "1", None, "missing"]}
    )

    result = _calculate_metrics(
        battles,
        (MetricSpec(metric="pairwise_win_rate", group_by=("group",)),),
    )

    values = [item["group"] for item in result["pairwise_win_rate"]["groups"]["group"]]
    assert {(type(value).__name__, value) for value in values} == {
        ("int", 1),
        ("str", "1"),
        ("str", "missing"),
        ("NoneType", None),
    }


def test_pairwise_win_rate_selects_and_orients_evaluation_rows():
    battles = pd.DataFrame(
        {
            "model_a": ["candidate", "opponent", "anchor-a"],
            "model_b": ["opponent", "candidate", "anchor-b"],
            "evaluation_model": ["candidate", "candidate", None],
            "pref": [0.0, 1.0, 1.0],
            "source": ["llm-judge", "llm-judge", "human"],
        }
    )

    result = PairwiseWinRateMetric().calculate(battles)

    assert result["num_battles"] == 2
    assert result["winrate"] == 1.0


def test_shared_registry_calculates_and_renders_point_bradley_terry():
    battles = pd.DataFrame(
        {
            "model_a": ["a", "a", "b", "b"],
            "model_b": ["b", "b", "a", "a"],
            "pref": [0.0, 0.0, 1.0, 1.0],
        }
    )

    results = _calculate_metrics(battles, (MetricSpec(metric="bradley_terry"),))

    ratings = results["bradley_terry"]["ratings"]
    assert set(ratings) == {"a", "b"}
    assert ratings["a"] > ratings["b"]
    rendered = render_metrics(results)
    assert "bradley_terry" in rendered
    assert "a:" in rendered
    assert "b:" in rendered


def test_length_controlled_winrate_accepts_elo_shaped_evaluation_battles():
    battles = pd.DataFrame(
        {
            "instruction_index": [0, 1, 2],
            "model_a": ["candidate", "opponent", "candidate"],
            "model_b": ["opponent", "candidate", "opponent"],
            "evaluation_model": ["candidate"] * 3,
            "completion_a": ["a", "bbbb", "aaaaaa"],
            "completion_b": ["bbb", "bb", "b"],
            "pref": [0.8, 0.2, 0.1],
            "orientation": ["single"] * 3,
        }
    )

    result = LengthControlledWinrateMetric().calculate(battles)

    assert result["num_pairs"] == 3
    assert result["num_scored"] == 3
    assert 0.0 <= result["winrate"] <= 1.0


def test_metric_builders_hide_registry_and_validate_parameters():
    assert available_metrics() == tuple(sorted(available_metrics()))
    assert "bradley_terry" in available_metrics()

    metric = build_metric("pairwise_win_rate")
    result = metric.calculate(pd.DataFrame({"pref": [0.1]}))
    assert result["winrate"] == 1.0
    assert "100.00%" in metric.render(result)

    with pytest.raises(ValueError, match="Unknown metric"):
        build_metric("missing")
    with pytest.raises(ValueError, match="unexpected keyword argument"):
        build_metric("pairwise_win_rate", {"soft": False})
    with pytest.raises(ValueError, match="soft must be a boolean"):
        build_metric("bradley_terry", {"soft": "false"})


def test_build_metrics_preserves_order_and_applies_overrides():
    requests = (
        MetricSpec(metric="pairwise_win_rate"),
        MetricSpec(metric="bradley_terry", parameters={"n_bootstraps": 2}),
    )

    configured = build_metrics(
        requests,
        parameter_overrides_by_metric={
            "bradley_terry": {"n_bootstraps": 3},
        },
    )

    assert [request.metric for request, _ in configured] == [
        "pairwise_win_rate",
        "bradley_terry",
    ]
    assert configured[1][1].n_bootstraps == 3
    assert configured[1][0].parameters == {"n_bootstraps": 2}
