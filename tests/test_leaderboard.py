"""Tests for the judgearena.leaderboard package (offline; judge calls mocked)."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from judgearena.config import EloArgs, JudgeArgs, PanelArgs, RunConfig
from judgearena.leaderboard.curate import build_panel, select_roster
from judgearena.leaderboard.kappa import language_kappa
from judgearena.leaderboard.panel import (
    PANEL_BATTLE_COLUMNS,
    Panel,
    load_panel,
    panel_hash,
    save_panel,
)
from judgearena.leaderboard.record import ResultRecord


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


def _curate_arena_df():
    # en: judge agrees with humans (kappa=1 -> kept). xx: judge disagrees (kappa<0 -> dropped).
    rows = []
    for i in range(5):  # 5 en battles, cap trims to n_per_language
        winner = "model_a" if i % 2 == 0 else "model_b"
        comp_a = "WIN" if winner == "model_a" else "LOSE"
        rows.append({
            "question_id": f"en{i}", "lang": "en", "model_a": "m1", "model_b": "m2",
            "winner": winner,
            "conversation_a": [{"role": "user", "content": f"en-q{i}"}, {"role": "assistant", "content": comp_a}],
            "conversation_b": [{"role": "user", "content": f"en-q{i}"}, {"role": "assistant", "content": "OPP"}],
        })
    for i in range(4):  # xx: judge says opposite of humans -> disagreement
        rows.append({
            "question_id": f"xx{i}", "lang": "xx", "model_a": "m1", "model_b": "m2",
            "winner": "model_a" if i % 2 == 1 else "model_b",
            "conversation_a": [{"role": "user", "content": f"xx-q{i}"}, {"role": "assistant", "content": "WIN" if i % 2 == 0 else "LOSE"}],
            "conversation_b": [{"role": "user", "content": f"xx-q{i}"}, {"role": "assistant", "content": "OPP"}],
        })
    return pd.DataFrame(rows)


def _fake_judge_prefs(judge_chat_model, instructions, completions_A, completions_B, **kwargs):
    # A wins (pref 0.0) iff completion_A is "WIN"; raw judge text encodes matching scores.
    prefs, annotations = [], []
    for c in completions_A:
        if c == "WIN":
            prefs.append(0.0)
            annotations.append(SimpleNamespace(judge_completion="score a: 9 score b: 1"))
        else:
            prefs.append(1.0)
            annotations.append(SimpleNamespace(judge_completion="score a: 1 score b: 9"))
    return annotations, None, pd.Series(prefs)


@pytest.fixture
def _patch_curate(monkeypatch):
    import judgearena.leaderboard.curate as cur
    monkeypatch.setattr(cur, "load_arena_dataframe", lambda arena: _curate_arena_df())
    monkeypatch.setattr(cur, "make_model", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(cur, "judge_and_parse_prefs", _fake_judge_prefs)
    monkeypatch.setattr(
        cur, "resolve_run_judge_prompt",
        lambda task, judge_cfg: SimpleNamespace(
            system_prompt=None, user_prompt_template=None, preset_name="x", parser_mode="score"
        ),
    )
    return cur


def test_build_panel_gates_languages_and_caps(_patch_curate):
    panel_args = PanelArgs(roster_models=["m1", "m2"], kappa_threshold=0.0, n_per_language=4)
    elo_args = EloArgs(arena=None, languages=["en", "xx"], elo_method="soft")
    panel = build_panel(panel_args, elo_args, judge_cfg=JudgeArgs(model="mock"), seed=0)
    assert set(panel.battles["lang"]) == {"en"}      # xx dropped (kappa <= 0)
    assert len(panel.battles) == 4                    # capped at n_per_language
    assert panel.meta["kappa_per_language"]["en"] == pytest.approx(1.0)
    assert "xx" not in panel.meta["languages_kept"]
    assert set(panel.battles["challenger_position"]) <= {"A", "B"}
    assert panel.battles["challenger_opponent"].isin(["m1", "m2"]).all()
    assert panel.meta["scorer"]["method"] == "soft"
    assert panel.meta["scorer"]["calibrated"] is False     # <10 pairs -> static T
    assert panel.meta["scorer"]["temperature"] == 0.3
    assert "mae_vs_human" in panel.meta
    assert list(panel.battles.columns) == list(PANEL_BATTLE_COLUMNS)


def test_build_panel_is_deterministic(_patch_curate):
    panel_args = PanelArgs(roster_models=["m1", "m2"], n_per_language=4)
    elo_args = EloArgs(languages=["en"], elo_method="soft")
    p1 = build_panel(panel_args, elo_args, judge_cfg=JudgeArgs(model="mock"), seed=0)
    p2 = build_panel(panel_args, elo_args, judge_cfg=JudgeArgs(model="mock"), seed=0)
    pd.testing.assert_frame_equal(
        p1.battles.reset_index(drop=True), p2.battles.reset_index(drop=True)
    )


def _toy_record():
    return ResultRecord(
        model="cand", panel_version="panel-v1", panel_hash="abc", judge_model="mock",
        elo_overall=1050.0, elo_std=12.0, elo_ci=[1030.0, 1075.0],
        elo_per_lang={"en": 1051.0}, winrate_overall=0.6, winrate_per_lang={"en": 0.6},
        n_battles=4, n_battles_per_lang={"en": 4}, kappa_per_lang={"en": 1.0},
        mae_vs_human=8.0, scorer={"mode": "pair_score", "temperature": 0.3},
        generation_params={"max_out_tokens": 32768}, seed=0,
        battles=pd.DataFrame({"battle_id": ["b1"], "judge_pref": [0.2]}),
    )


def test_resultrecord_to_dict_keys():
    d = _toy_record().to_dict()
    assert "battles" not in d
    assert set(d) == {
        "model", "panel_version", "panel_hash", "judge_model", "elo_overall",
        "elo_std", "elo_ci", "elo_per_lang", "winrate_overall", "winrate_per_lang",
        "n_battles", "n_battles_per_lang", "kappa_per_lang", "mae_vs_human",
        "scorer", "generation_params", "seed", "schema_version", "submitter",
        "created_utc",
    }


def test_resultrecord_save_writes_both_files(tmp_path):
    out = _toy_record().save(tmp_path / "cand")
    assert (out / "result.json").exists()
    assert (out / "battles.parquet").exists()
    reloaded = json.loads((out / "result.json").read_text())
    assert reloaded["elo_overall"] == 1050.0
    assert pd.read_parquet(out / "battles.parquet")["battle_id"].tolist() == ["b1"]
