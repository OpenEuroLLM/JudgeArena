"""Tests for the judgearena.leaderboard package (offline; judge calls mocked)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from judgearena.config import EloArgs, PanelArgs, RunConfig
from judgearena.leaderboard.curate import select_roster
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


def test_eloargs_elo_method_and_compat_properties():
    e = EloArgs()
    assert e.elo_method == "soft"          # matches old soft_elo=True, calibrate=False
    assert e.soft_elo is True              # derived property
    assert e.calibrate_temperature is False
    assert e.soft_elo_temperature == 0.3
    assert e.n_bootstraps == 20
    assert e.calibration_size is None
    assert e.arena is None
    assert e.baseline_model is None


def test_eloargs_method_drives_properties():
    assert EloArgs(elo_method="hard").soft_elo is False
    cs = EloArgs(elo_method="calibrated_soft")
    assert cs.soft_elo is True
    assert cs.calibrate_temperature is True
    # soft_elo / calibrate_temperature are no longer config fields
    assert "soft_elo" not in EloArgs.model_fields
    assert "calibrate_temperature" not in EloArgs.model_fields
    assert "elo_method" in EloArgs.model_fields


def test_panel_args_defaults_and_runconfig_field():
    p = PanelArgs()
    assert p.panel_version == "panel-v1"
    assert p.roster_min_annotations == 100
    assert p.roster_min_languages == 15
    assert p.roster_max_models == 20
    assert p.kappa_threshold == 0.0
    assert p.n_per_language == 100
    assert p.roster_models is None
    assert p.panel_dir == "panels"
    assert "panel" in RunConfig.model_fields


def _roster_df():
    # strong appears in 2 langs x 2 battles = 4 appearances; weak in 1 lang once
    rows = []
    for lang in ["en", "fr"]:
        for _ in range(2):
            rows.append({"model_a": "strong", "model_b": "mid", "lang": lang})
    rows.append({"model_a": "weak", "model_b": "mid", "lang": "en"})
    return pd.DataFrame(rows)


def test_select_roster_applies_thresholds():
    args = PanelArgs(roster_min_annotations=4, roster_min_languages=2)
    # strong: 4 appearances / 2 langs -> qualifies. mid: 5 appearances but...
    # mid appears in en+fr -> 2 langs, 5 appearances -> also qualifies.
    roster = select_roster(_roster_df(), args)
    assert set(roster) == {"strong", "mid"}
    assert "weak" not in roster


def test_select_roster_respects_max_models_and_explicit_override():
    args = PanelArgs(roster_min_annotations=1, roster_min_languages=1, roster_max_models=1)
    roster = select_roster(_roster_df(), args)
    assert roster == ["mid"]  # highest appearance count

    explicit = PanelArgs(roster_models=["x", "y"])
    assert select_roster(_roster_df(), explicit) == ["x", "y"]
