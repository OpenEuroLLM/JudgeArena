"""Tests for the judgearena.leaderboard package (offline; judge calls mocked)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from judgearena.leaderboard.kappa import language_kappa
from judgearena.leaderboard.panel import Panel, load_panel, panel_hash, save_panel


def test_language_kappa_perfect_agreement():
    judge = pd.Series([0.0, 1.0, 0.0, 1.0])
    human = pd.Series([0.0, 1.0, 0.0, 1.0])
    assert language_kappa(judge, human) == pytest.approx(1.0)


def test_language_kappa_chance_agreement_is_zero():
    judge = pd.Series([0.0, 0.0, 1.0, 1.0])
    human = pd.Series([0.0, 1.0, 0.0, 1.0])
    assert language_kappa(judge, human) == pytest.approx(0.0)


def test_language_kappa_excludes_ties_and_nan():
    # last two rows (a tie and a NaN) must be dropped, leaving perfect agreement
    judge = pd.Series([0.0, 1.0, 0.0, 1.0, 0.5, 1.0])
    human = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, float("nan")])
    assert language_kappa(judge, human) == pytest.approx(1.0)


def test_language_kappa_single_class_returns_nan():
    judge = pd.Series([0.0, 0.0, 0.0])
    human = pd.Series([0.0, 0.0, 0.0])
    assert math.isnan(language_kappa(judge, human))


def _toy_battles():
    return pd.DataFrame(
        {
            "battle_id": ["b1", "b2"],
            "lang": ["en", "fr"],
            "model_a": ["m1", "m1"],
            "model_b": ["m2", "m2"],
            "instruction": ["q1", "q2"],
            "completion_a": ["ca1", "ca2"],
            "completion_b": ["cb1", "cb2"],
            "human_winner": ["model_a", "model_b"],
            "judge_pref": [0.1, 0.8],
            "judge_pref_hard": [0.0, 1.0],
            "challenger_opponent": ["m1", "m2"],
            "challenger_position": ["A", "B"],
        }
    )


def test_panel_hash_is_order_independent_and_change_sensitive():
    b = _toy_battles()
    h1 = panel_hash(b)
    h2 = panel_hash(b.iloc[::-1].reset_index(drop=True))
    assert h1 == h2
    changed = b.copy()
    changed.loc[0, "judge_pref"] = 0.999
    assert panel_hash(changed) != h1


def test_save_load_panel_round_trip(tmp_path):
    meta = {"panel_version": "panel-v1", "judge_model": "mock", "languages_kept": ["en", "fr"]}
    panel = Panel(meta=meta, battles=_toy_battles())
    out = save_panel(panel, tmp_path / "panel-v1")
    assert (out / "battles.parquet").exists()
    assert (out / "panel.json").exists()

    loaded = load_panel(out)
    assert loaded.meta["panel_version"] == "panel-v1"
    assert loaded.meta["panel_hash"] == panel_hash(_toy_battles())
    pd.testing.assert_frame_equal(
        loaded.battles.reset_index(drop=True), _toy_battles(), check_like=True
    )
