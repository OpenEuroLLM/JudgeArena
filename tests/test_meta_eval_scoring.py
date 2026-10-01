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
from judgearena.benchmarks.scoring import build_metric


def _rows(specs: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        specs, columns=["battle_id", "model_a", "model_b", "reference_pref", "pref"]
    ).assign(sampled=True)


def _balanced_battles():
    return _rows(
        [
            (f"{a}{b}-{i}", a, b, float(i), float(i))
            for a, b in (("a", "b"), ("a", "c"), ("b", "c"))
            for i in range(2)
        ]
    )


def _ranking_metric(*, include_human_ties=False):
    return MetaEvalRankingMetric(
        n_bootstraps=0, tie_tolerance=0.01, include_human_ties=include_human_ties
    )


def _elo_gap_metric(
    battle_counts: tuple[int, ...], n_seeds: int
) -> MetaEvalEloGapMetric:
    return MetaEvalEloGapMetric(
        battle_counts=battle_counts, n_seeds=n_seeds, tie_tolerance=0.01
    )


def test_agreement_reports_missing_judgments_and_excludes_only_human_ties():
    battles = _rows(
        [
            ("1", "a", "b", 0.0, 0.0),
            ("2", "b", "c", 1.0, 0.5),
            ("3", "a", "c", 0.5, 0.5),
            ("4", "a", "b", 0.0, np.nan),
            ("5", "b", "c", 1.0, np.nan),
            ("unsampled", "a", "b", 0.0, np.nan),
        ]
    )
    battles.loc[5, "sampled"] = False
    result = MetaEvalAgreementMetric(n_bootstraps=8, tie_tolerance=0.01).calculate(
        battles, rng=np.random.default_rng(7)
    )

    all_rows = result["all"]
    assert (all_rows["n_attempted"], all_rows["n_complete"]) == (5, 3)
    assert all_rows["coverage"] == pytest.approx(3 / 5)
    assert all_rows["accuracy_attempted"] == pytest.approx(2 / 5)
    assert all_rows["accuracy_complete"] == pytest.approx(2 / 3)
    assert all_rows["cohen_kappa"] == pytest.approx(0.5)
    assert 0 < all_rows["accuracy_complete_bootstraps_valid"] <= 8
    assert 0 < all_rows["accuracy_complete_se"] < 1
    no_ties = result["no_human_ties"]
    assert (no_ties["n_attempted"], no_ties["n_complete"]) == (4, 2)
    assert no_ties["accuracy_attempted"] == pytest.approx(1 / 4)
    assert no_ties["accuracy_complete"] == pytest.approx(1 / 2)
    assert no_ties["cohen_kappa"] == pytest.approx(1 / 3)


def test_ranking_surfaces_unexpected_fit_errors(monkeypatch):
    def fail_fit(*args, **kwargs):
        raise ValueError("unexpected fit failure")

    monkeypatch.setattr(scoring_module, "fit_bradley_terry", fail_fit)
    with pytest.raises(ValueError, match="unexpected fit failure"):
        _ranking_metric().calculate(_balanced_battles())


def test_ranking_human_ties_are_configurable_and_model_set_stays_fixed():
    battles = _rows([("1", "a", "b", 0.0, 0.0), ("2", "a", "b", 0.5, 0.5)])
    included = _ranking_metric(include_human_ties=True).calculate(battles)
    assert included["hard"]["spearman"] == pytest.approx(1.0)
    assert included["soft"]["elo_mae"] == pytest.approx(0.0)

    battles = pd.concat([battles, _rows([("tie-edge", "a", "c", 0.5, 0.0)])])
    excluded = _ranking_metric().calculate(battles)
    included = _ranking_metric(include_human_ties=True).calculate(battles)
    assert excluded["n_models"] == 3
    assert excluded["n_battles"] == 1
    assert math.isnan(excluded["hard"]["elo_mae"])
    assert included["n_battles"] == 3
    assert math.isfinite(included["hard"]["elo_mae"])


def test_malformed_battles_raise_instead_of_becoming_unavailable():
    battles = _balanced_battles()
    battles.loc[0, "pref"] = 1.1
    with pytest.raises(ValueError, match="numeric preferences"):
        _ranking_metric().calculate(battles)


def test_elo_gap_configuration_requires_increasing_budgets():
    with pytest.raises(ValueError, match="Invalid parameters"):
        build_metric(
            "meta_eval_elo_gap",
            {"battle_counts": [2, 1], "n_seeds": 2, "tie_tolerance": 0.01},
        )


def test_elo_gap_distinguishes_soft_confidence_from_hard_labels():
    battles = _balanced_battles()
    metric = _elo_gap_metric((4,), 3)
    result = metric.calculate(battles, rng=np.random.default_rng(11))
    assert result["soft"] == result["hard"]
    assert result["hard"][0]["mean_gap"] == pytest.approx(0.0, abs=1e-6)
    assert result["hard"][0]["n_seeds_valid"] == 3

    battles.loc[battles["pref"].eq(0.0), "pref"] = 0.2
    softened = metric.calculate(battles, rng=np.random.default_rng(11))
    assert softened["hard"] == result["hard"]
    assert softened["soft"][0]["mean_gap"] != pytest.approx(
        result["hard"][0]["mean_gap"]
    )


def test_elo_gap_counts_attempts_before_parse_filtering():
    battles = _balanced_battles()
    battles.loc[battles["battle_id"].eq("ab-0"), "pref"] = np.nan
    result = _elo_gap_metric((1, 4), 4).calculate(battles, rng=np.random.default_rng(7))

    rows = result["soft"]
    assert rows[0]["mean_complete_per_model"] <= rows[1]["mean_complete_per_model"]
    assert rows[1]["attempted_battles_per_model"] == 4
    assert rows[1]["mean_complete_per_model"] == pytest.approx(10 / 3)
    assert rows[1]["mean_used_per_model"] == pytest.approx(10 / 3)


def test_elo_gap_keeps_fixed_model_set_and_fails_whole_replicates():
    battles = _balanced_battles()
    focal_a = battles["model_a"].eq("a") | battles["model_b"].eq("a")
    battles.loc[focal_a, "pref"] = np.nan
    result = _elo_gap_metric((4,), 2).calculate(battles, rng=np.random.default_rng(3))

    assert result["n_models"] == 3
    assert result["soft"][0]["n_seeds_valid"] == 0
    assert math.isnan(result["soft"][0]["mean_gap"])


def test_elo_gap_warns_and_returns_empty_for_attempted_battle_shortfalls(caplog):
    battles = _balanced_battles()
    battles.loc[battles["battle_id"].eq("ab-0"), "sampled"] = False
    battles.loc[battles["battle_id"].eq("ab-0"), "pref"] = np.nan

    result = _elo_gap_metric((4,), 1).calculate(battles, rng=np.random.default_rng(1))

    assert result == {}
    assert "Skipping meta_eval_elo_gap" in caplog.text
    assert "at least 4 attempted incident battles" in caplog.text
