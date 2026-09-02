"""Behavior tests for composable pairwise metrics."""

import math

import numpy as np
import pandas as pd
import pytest

from judgearena.benchmarks.pairwise.scoring import (
    calculate_metrics,
    collapse_pairwise_battles,
    length_controlled_winrate,
    pairwise_win_rate,
)
from judgearena.tasks.schema import MetricSpec


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
    battles = pd.DataFrame({"pref": [0.0, 0.0, 1.0, None]})

    result = pairwise_win_rate(battles)

    assert result == {
        "num_battles": 4,
        "winrate": pytest.approx(2 / 3),
        "num_wins": 2,
        "num_losses": 1,
        "num_ties": 0,
        "num_missing": 1,
    }


def test_pairwise_win_rate_preserves_soft_preferences():
    result = pairwise_win_rate(pd.DataFrame({"pref": [0.1, 0.3]}))

    assert result["winrate"] == pytest.approx(0.8)


def test_length_controlled_winrate_predicts_at_equal_length(monkeypatch):
    from judgearena.benchmarks.pairwise.scoring import metrics

    monkeypatch.setattr(metrics, "BOOTSTRAP_ROUNDS", 100)
    differences = np.arange(-100, 101, 10, dtype=float)
    scale = differences.std(ddof=1)
    expected = 0.4
    intercept = math.log(expected / (1 - expected))
    outcomes = 1 / (1 + np.exp(-(intercept + 0.8 * differences / scale)))

    result = length_controlled_winrate(_battle_rows(differences, outcomes))

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
    result = length_controlled_winrate(battles)

    assert collapsed["n_parsed"].tolist() == [2, 1, 2]
    assert result["num_pairs"] == 3
    assert result["num_scored"] == 2
    assert result["winrate"] is None


def test_metrics_can_be_calculated_by_group():
    battles = pd.DataFrame(
        {
            "pref": [0.0, 1.0, 0.0, 0.0],
            "category": ["a", "a", "b", "b"],
        }
    )

    results = calculate_metrics(
        battles,
        (MetricSpec(metric="pairwise_win_rate", group_by=("category",)),),
    )

    assert results == {
        "pairwise_win_rate": {
            "num_battles": 4,
            "winrate": 0.75,
            "num_wins": 3,
            "num_losses": 1,
            "num_ties": 0,
            "num_missing": 0,
            "groups": {
                "category": [
                    {
                        "group": "a",
                        "values": {
                            "num_battles": 2,
                            "winrate": 0.5,
                            "num_wins": 1,
                            "num_losses": 1,
                            "num_ties": 0,
                            "num_missing": 0,
                        },
                    },
                    {
                        "group": "b",
                        "values": {
                            "num_battles": 2,
                            "winrate": 1.0,
                            "num_wins": 2,
                            "num_losses": 0,
                            "num_ties": 0,
                            "num_missing": 0,
                        },
                    },
                ]
            },
        }
    }


def test_pairwise_metrics_reject_invalid_preferences():
    with pytest.raises(ValueError, match="finite values"):
        pairwise_win_rate(pd.DataFrame({"pref": [0.0, float("inf")]}))


def test_empty_length_controlled_input_is_reported():
    battles = _battle_rows(np.array([], dtype=float), np.array([], dtype=float))

    result = length_controlled_winrate(battles)

    assert result == {
        "num_pairs": 0,
        "num_scored": 0,
        "winrate": None,
        "reason": "not_enough_complete_pairs",
    }


def test_bootstrap_does_not_replace_undefined_draws(monkeypatch):
    from judgearena.benchmarks.pairwise.scoring import metrics

    monkeypatch.setattr(metrics, "BOOTSTRAP_ROUNDS", 100)
    differences = np.array([-10, 0, 10], dtype=float)
    outcomes = np.array([0.0, 1.0, 0.0])

    result = length_controlled_winrate(_battle_rows(differences, outcomes))

    assert result["winrate"] is not None
    assert result["confidence_interval"] is None


def test_grouped_metric_rejects_missing_column():
    with pytest.raises(ValueError, match="missing column 'category'"):
        calculate_metrics(
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


def test_collapse_rejects_direct_rows_without_reversed_pairs():
    battles = _battle_rows(np.array([0]), np.array([0.5]))
    battles["orientation"] = "direct"

    with pytest.raises(ValueError, match="complete direct/reversed pairs"):
        collapse_pairwise_battles(battles)


def test_collapse_rejects_different_completions_across_orders():
    direct = _battle_rows(np.array([0]), np.array([0.5]))
    direct["orientation"] = "direct"
    reversed_row = direct.copy()
    reversed_row["orientation"] = "reversed"
    reversed_row["completion_model"] = "different"

    with pytest.raises(ValueError, match="different completions"):
        collapse_pairwise_battles(pd.concat([direct, reversed_row]))


def test_length_controlled_winrate_reports_mixed_baselines():
    battles = _battle_rows(np.array([-10, 0, 10]), np.array([0.2, 0.5, 0.8]))
    battles.loc[2, "baseline"] = "other-reference"

    result = length_controlled_winrate(battles)

    assert result["winrate"] is None
    assert result["reason"] == "multiple_baseline_models"


def test_grouped_metrics_preserve_distinct_group_values():
    battles = pd.DataFrame(
        {"pref": [0.0, 0.0, 0.0, 0.0], "group": [1, "1", None, "missing"]}
    )

    result = calculate_metrics(
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
