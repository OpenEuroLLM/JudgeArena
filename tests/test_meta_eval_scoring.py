"""Focused tests for meta-evaluation agreement and ranking metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import judgearena.benchmarks.meta_eval.scoring as scoring_module
from judgearena.benchmarks.meta_eval.scoring import (
    MetaEvalAgreementMetric,
    MetaEvalEloGapMetric,
    MetaEvalRankingMetric,
)
from judgearena.benchmarks.scoring import available_metrics, build_metric


def _rows(specs: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "battle_id": battle_id,
                "model_a": model_a,
                "model_b": model_b,
                "reference_pref": reference,
                "pref": judge,
                "sampled": True,
            }
            for battle_id, model_a, model_b, reference, judge in specs
        ]
    )


def _agreement_metric(n_bootstraps: int = 0) -> MetaEvalAgreementMetric:
    return MetaEvalAgreementMetric(n_bootstraps=n_bootstraps, tie_tolerance=0.01)


def _ranking_metric(
    n_bootstraps: int = 0, *, include_human_ties: bool = False
) -> MetaEvalRankingMetric:
    return MetaEvalRankingMetric(
        n_bootstraps=n_bootstraps,
        tie_tolerance=0.01,
        include_human_ties=include_human_ties,
    )


def _elo_gap_metric(
    battle_counts: tuple[int, ...], n_seeds: int
) -> MetaEvalEloGapMetric:
    return MetaEvalEloGapMetric(
        battle_counts=battle_counts, n_seeds=n_seeds, tie_tolerance=0.01
    )


def _ranking_battles() -> pd.DataFrame:
    rows = []
    preferences = {
        ("a", "b"): ([0.0, 0.0, 0.0, 1.0], [0.49, 0.49, 0.49, 1.0]),
        ("a", "c"): ([0.0, 0.0, 0.0, 1.0], [0.01, 0.01, 0.01, 1.0]),
        ("b", "c"): ([0.0, 0.0, 0.0, 1.0], [0.0, 0.51, 0.51, 0.51]),
    }
    for (model_a, model_b), (human, judge) in preferences.items():
        for index, (reference_pref, pref) in enumerate(zip(human, judge, strict=True)):
            rows.append(
                {
                    "battle_id": f"{model_a}{model_b}-{index}",
                    "model_a": model_a,
                    "model_b": model_b,
                    "reference_pref": reference_pref,
                    "pref": pref,
                    "sampled": True,
                }
            )
    return pd.DataFrame(rows)


def test_meta_eval_metrics_are_registered_as_configured_metrics():
    assert {
        "meta_eval_agreement",
        "meta_eval_elo_gap",
        "meta_eval_ranking",
    } <= set(available_metrics())
    assert isinstance(
        build_metric("meta_eval_agreement", {"n_bootstraps": 0, "tie_tolerance": 0.01}),
        MetaEvalAgreementMetric,
    )
    assert isinstance(
        build_metric(
            "meta_eval_ranking",
            {
                "n_bootstraps": 0,
                "tie_tolerance": 0.01,
                "include_human_ties": False,
            },
        ),
        MetaEvalRankingMetric,
    )
    assert isinstance(
        build_metric(
            "meta_eval_elo_gap",
            {"battle_counts": [1, 2], "n_seeds": 2, "tie_tolerance": 0.01},
        ),
        MetaEvalEloGapMetric,
    )


def test_agreement_reports_attempted_missing_and_numeric_kappa_semantics():
    battles = _rows(
        [
            ("1", "a", "b", 0.0, 0.0),
            ("2", "b", "c", 1.0, 0.0),
            ("3", "a", "c", 0.5, 0.5),
            ("4", "a", "b", 0.0, np.nan),
            ("5", "b", "c", 1.0, np.nan),
        ]
    )

    battles.loc[len(battles)] = {
        "battle_id": "unsampled",
        "model_a": "a",
        "model_b": "b",
        "reference_pref": 0.0,
        "pref": np.nan,
        "sampled": False,
    }
    result = _agreement_metric().calculate(battles)

    assert result["all"] == {
        "n_attempted": 5,
        "n_complete": 3,
        "coverage": pytest.approx(0.6),
        "accuracy_attempted": pytest.approx(0.4),
        "accuracy_complete": pytest.approx(2 / 3),
        "cohen_kappa": pytest.approx(0.5),
        "accuracy_attempted_se": pytest.approx(float("nan"), nan_ok=True),
        "accuracy_complete_se": pytest.approx(float("nan"), nan_ok=True),
        "accuracy_complete_bootstraps_valid": 0,
        "cohen_kappa_se": pytest.approx(float("nan"), nan_ok=True),
        "n_bootstraps_requested": 0,
        "n_kappa_bootstraps_valid": 0,
    }
    no_ties = result["no_human_ties"]
    assert no_ties["n_attempted"] == 4
    assert no_ties["n_complete"] == 2
    assert no_ties["coverage"] == pytest.approx(0.5)
    assert no_ties["accuracy_attempted"] == pytest.approx(0.25)
    assert no_ties["accuracy_complete"] == pytest.approx(0.5)
    assert no_ties["cohen_kappa"] == pytest.approx(0.0)


def test_agreement_bootstrap_is_row_order_invariant_and_reports_finite_draws():
    battles = _rows(
        [
            ("1", "a", "b", 0.0, 0.0),
            ("2", "b", "c", 1.0, 0.0),
            ("3", "a", "c", 0.0, np.nan),
        ]
    )
    metric = _agreement_metric(8)
    result = metric.calculate(battles, rng=np.random.default_rng(7))["all"]
    reordered = metric.calculate(battles.iloc[::-1], rng=np.random.default_rng(7))[
        "all"
    ]

    expected_rng = np.random.default_rng(7)
    complete_accuracies = []
    outcomes = np.array([1.0, 0.0, np.nan])
    for _ in range(8):
        sampled = outcomes[expected_rng.integers(0, 3, size=3)]
        complete = sampled[np.isfinite(sampled)]
        if len(complete):
            complete_accuracies.append(float(complete.mean()))

    assert result == pytest.approx(reordered, nan_ok=True)
    assert result["accuracy_complete_bootstraps_valid"] == len(complete_accuracies)
    assert result["accuracy_complete_se"] == pytest.approx(
        np.std(complete_accuracies, ddof=1)
    )


def test_no_human_ties_drops_only_reference_ties_and_keeps_judge_ties():
    battles = _rows(
        [
            ("1", "a", "b", 0.5, 0.0),
            ("2", "b", "c", 1.0, 0.5),
            ("3", "a", "c", 0.0, 0.0),
        ]
    )

    view = _agreement_metric().calculate(battles)["no_human_ties"]

    assert view["n_attempted"] == 2
    assert view["accuracy_attempted"] == pytest.approx(0.5)


def test_ranking_preserves_distinct_hard_and_soft_results():
    point = _ranking_metric().calculate(_ranking_battles())

    assert math.isfinite(point["hard"]["spearman"])
    assert math.isfinite(point["soft"]["elo_mae"])
    assert point["hard"]["elo_mae"] != pytest.approx(point["soft"]["elo_mae"])

    bootstrapped = _ranking_metric(3).calculate(
        _ranking_battles(), rng=np.random.default_rng(4)
    )
    assert bootstrapped["n_bootstraps_requested"] == 3
    assert 0 <= bootstrapped["n_bootstraps_valid"] <= 3


def test_ranking_surfaces_unexpected_fit_errors(monkeypatch):
    def fail_fit(*args, **kwargs):
        raise ValueError("unexpected fit failure")

    monkeypatch.setattr(scoring_module, "fit_bradley_terry", fail_fit)
    with pytest.raises(ValueError, match="unexpected fit failure"):
        _ranking_metric().calculate(_ranking_battles())


def test_ranking_is_invariant_to_row_order_and_global_ab_swap():
    battles = _ranking_battles()
    shuffled = battles.sample(frac=1, random_state=7)
    swapped = battles.copy()
    swapped[["model_a", "model_b"]] = swapped[["model_b", "model_a"]]
    swapped["reference_pref"] = 1.0 - swapped["reference_pref"]
    swapped["pref"] = 1.0 - swapped["pref"]
    metric = _ranking_metric(8)

    expected = metric.calculate(battles, rng=np.random.default_rng(9))
    reordered = metric.calculate(shuffled, rng=np.random.default_rng(9))
    reversed_ab = metric.calculate(swapped, rng=np.random.default_rng(9))

    for result in (reordered, reversed_ab):
        assert result["n_bootstraps_valid"] == expected["n_bootstraps_valid"]
        for kind in ("hard", "soft"):
            for value in ("spearman", "spearman_se", "elo_mae", "elo_mae_se"):
                assert result[kind][value] == pytest.approx(
                    expected[kind][value], nan_ok=True, abs=1e-5
                )


def test_ranking_human_ties_are_configurable_and_model_set_stays_fixed():
    battles = _ranking_battles()
    tie = battles.iloc[[0]].copy()
    tie["battle_id"] = "only-c-edge"
    tie["model_a"] = "a"
    tie["model_b"] = "d"
    tie["reference_pref"] = 0.5
    tie["pref"] = 0.0
    battles = pd.concat([battles, tie], ignore_index=True)

    excluded = _ranking_metric().calculate(battles)
    included = _ranking_metric(0, include_human_ties=True).calculate(battles)

    assert excluded["n_models"] == 4
    assert excluded["n_battles"] == 12
    assert math.isnan(excluded["hard"]["elo_mae"])
    assert included["n_battles"] == 13
    assert math.isfinite(included["hard"]["elo_mae"])


def test_ranking_returns_numeric_unavailable_for_disconnected_graphs():
    disconnected = _rows(
        [
            ("1", "a", "b", 0.0, 0.0),
            ("2", "a", "b", 1.0, 1.0),
            ("3", "c", "d", 0.0, 0.0),
            ("4", "c", "d", 1.0, 1.0),
        ]
    )

    result = _ranking_metric().calculate(disconnected)

    assert result["n_bootstraps_valid"] == 0
    assert math.isnan(result["hard"]["spearman"])
    assert math.isnan(result["soft"]["elo_mae"])


def test_ranking_supports_connected_two_model_inputs():
    battles = _rows(
        [
            ("1", "a", "b", 0.0, 0.0),
            ("2", "a", "b", 0.5, 0.5),
        ]
    )

    result = _ranking_metric(0, include_human_ties=True).calculate(battles)

    assert result["n_models"] == 2
    assert result["hard"]["spearman"] == pytest.approx(1.0)
    assert result["soft"]["elo_mae"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("pref", 1.1, "numeric preferences"),
        ("reference_pref", 0.25, "must be 0, 0.5, or 1"),
        ("model_a", None, "non-null strings"),
        ("model_b", "a", "self-comparisons"),
    ],
)
def test_malformed_battles_raise_instead_of_becoming_unavailable(
    column, value, message
):
    battles = _ranking_battles()
    battles.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        _ranking_metric().calculate(battles)


def _elo_gap_battles() -> pd.DataFrame:
    specs = []
    for model_a, model_b in (("a", "b"), ("a", "c"), ("b", "c")):
        specs.extend(
            [
                (f"{model_a}{model_b}-0", model_a, model_b, 0.0, 0.0),
                (f"{model_a}{model_b}-1", model_a, model_b, 1.0, 1.0),
            ]
        )
    return _rows(specs)


@pytest.mark.parametrize(
    "parameters",
    [
        {"battle_counts": []},
        {"battle_counts": [0]},
        {"battle_counts": [2, 1]},
        {"battle_counts": [1, 1]},
        {"n_seeds": 0},
        {"tie_tolerance": 0.5},
    ],
)
def test_elo_gap_configuration_rejects_invalid_parameters(parameters):
    valid = {
        "battle_counts": [1, 2],
        "n_seeds": 2,
        "tie_tolerance": 0.01,
    }
    with pytest.raises(ValueError, match="Invalid parameters"):
        build_metric("meta_eval_elo_gap", valid | parameters)


def test_elo_gap_surfaces_unexpected_fit_errors(monkeypatch):
    def fail_fit(*args, **kwargs):
        raise TypeError("unexpected Elo fit failure")

    monkeypatch.setattr(scoring_module, "fit_bradley_terry", fail_fit)
    with pytest.raises(TypeError, match="unexpected Elo fit failure"):
        _elo_gap_metric((1,), 1).calculate(
            _elo_gap_battles(), rng=np.random.default_rng(11)
        )


def test_elo_gap_bundles_shared_methods_and_is_row_order_invariant():
    battles = _elo_gap_battles()
    battles.loc[0, ["reference_pref", "pref"]] = 0.5
    metric = _elo_gap_metric((1, 2, 4), 3)

    with pytest.raises(ValueError, match="requires an RNG"):
        metric.calculate(battles)
    result = metric.calculate(battles, rng=np.random.default_rng(11))
    shuffled = metric.calculate(
        battles.sample(frac=1, random_state=4), rng=np.random.default_rng(11)
    )

    assert result == shuffled
    assert result["n_models"] == 3
    assert result["soft"] == result["hard"]
    for variant in ("hard", "soft"):
        full_budget = result[variant][-1]
        assert full_budget["mean_gap"] == pytest.approx(0.0, abs=1e-6)
        assert full_budget["n_seeds_valid"] == 3
        assert full_budget["mean_used_per_model"] == 4


def test_elo_gap_draws_attempts_before_parse_filtering_and_uses_nested_prefixes():
    battles = _elo_gap_battles()
    battles.loc[battles["battle_id"].eq("ab-0"), "pref"] = np.nan
    result = _elo_gap_metric((1, 2, 4), 4).calculate(
        battles, rng=np.random.default_rng(7)
    )

    for variant in ("hard", "soft"):
        rows = result[variant]
        complete = [row["mean_complete_per_model"] for row in rows]
        assert complete == sorted(complete)
        assert rows[-1]["attempted_battles_per_model"] == 4
        assert rows[-1]["mean_complete_per_model"] == pytest.approx(10 / 3)
        assert rows[-1]["mean_used_per_model"] == pytest.approx(10 / 3)


def test_elo_gap_keeps_fixed_model_set_and_fails_whole_replicates():
    battles = _elo_gap_battles()
    focal_a = battles["model_a"].eq("a") | battles["model_b"].eq("a")
    battles.loc[focal_a, "pref"] = np.nan
    result = _elo_gap_metric((4,), 2).calculate(battles, rng=np.random.default_rng(3))

    assert result["n_models"] == 3
    for variant in ("hard", "soft"):
        row = result[variant][0]
        assert row["n_seeds_valid"] == 0
        assert math.isnan(row["mean_gap"])


def test_elo_gap_rejects_attempted_battle_shortfalls():
    battles = _elo_gap_battles()
    battles.loc[battles["battle_id"].eq("ab-0"), "sampled"] = False
    battles.loc[battles["battle_id"].eq("ab-0"), "pref"] = np.nan

    with pytest.raises(ValueError, match="Every model needs at least 4"):
        _elo_gap_metric((4,), 1).calculate(battles, rng=np.random.default_rng(1))
