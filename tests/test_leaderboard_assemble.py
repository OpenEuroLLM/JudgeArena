# tests/test_leaderboard_assemble.py
import inspect

import pandas as pd

from judgearena.leaderboard.assemble import (
    assemble_bundle,
    assemble_scores,
)

PANEL_META = {
    "panel_version": "v1",
    "panel_hash": "abc",
    "judge_model": "j",
    "baseline_model": "m1",
    "mae_vs_human": 1.0,
    "kappa_per_language": {"en": 0.7},
    "scorer": {"method": "soft"},
}
ANCHOR_RATINGS = {
    "overall": {"m1": 0.0, "m2": 50.0},
    "per_lang": {"en": {"m1": 0.0, "m2": 40.0}},
    "counts_overall": {"m1": 10, "m2": 10},
    "counts_per_lang": {"en": {"m1": 5, "m2": 5}},
}
CALIBRATION = {"mae": 1.0, "spearman": 0.9, "points": [{"model": "m1", "human_elo": 0.0, "judge_elo": 1.0, "judge_ci": [0.0, 2.0]}]}
ANCHOR_H2H = {"pairwise": {"m1": {"m2": [4.0, 10]}, "m2": {"m1": [6.0, 10]}}}


def _record():
    return {
        "model": "sub",
        "tag": None,
        "panel_hash": "abc",
        "elo_overall": 100.0,
        "elo_ci": [90.0, 110.0],
        "elo_per_lang": {"en": 80.0},
        "n_battles": 6,
        "n_battles_per_lang": {"en": 6},
        "winrate_overall": 0.6,
        "winrate_per_lang": {"en": 0.6},
    }


def _record_battles():
    return pd.DataFrame(
        {
            "lang": ["en", "en"],
            "opponent": ["m1", "m2"],
            "position": ["A", "A"],
            "judge_pref": [0.1, 0.2],  # <0.5 at position A => submission wins
        }
    )


def test_assemble_rows_rank_submission_top():
    bundle = assemble_bundle(
        PANEL_META, ANCHOR_RATINGS, CALIBRATION, ANCHOR_H2H,
        [_record()], {"sub": _record_battles()},
    )
    assert [r["model"] for r in bundle["rows"]] == ["sub", "m2", "m1"]
    sub_row = bundle["rows"][0]
    assert sub_row["is_submission"] is True
    assert sub_row["winrate"] == 0.6
    assert sub_row["ci_low"] == 90.0


def test_assemble_calibration_passthrough():
    bundle = assemble_bundle(PANEL_META, ANCHOR_RATINGS, CALIBRATION, ANCHOR_H2H, [], {})
    assert bundle["calibration"] is CALIBRATION
    assert bundle["languages"] == ["en"]


def test_assemble_h2h_extends_anchor_block():
    bundle = assemble_bundle(
        PANEL_META, ANCHOR_RATINGS, CALIBRATION, ANCHOR_H2H,
        [_record()], {"sub": _record_battles()},
    )
    h2h = bundle["head_to_head"]
    assert set(h2h["models"]) == {"m1", "m2", "sub"}
    i_sub = h2h["models"].index("sub")
    i_m1 = h2h["models"].index("m1")
    # submission beat m1 -> winrate 1.0, count 1
    assert h2h["counts"][i_sub][i_m1] == 1
    assert h2h["winrate"][i_sub][i_m1] == 1.0


def test_panel_hash_mismatch_skipped():
    rec = _record()
    rec["panel_hash"] = "WRONG"
    bundle = assemble_bundle(PANEL_META, ANCHOR_RATINGS, CALIBRATION, ANCHOR_H2H, [rec], {"sub": _record_battles()})
    assert all(not r["is_submission"] for r in bundle["rows"])


def test_assemble_scores_longform():
    scores = assemble_scores([_record()], {"sub": _record_battles()}, "abc")
    assert list(scores.columns) == ["model", "tag", "lang", "judge_pref"]
    assert len(scores) == 2
    assert set(scores["model"]) == {"sub"}


def test_assemble_scores_skips_panel_hash_mismatch():
    rec = _record()
    rec["panel_hash"] = "WRONG"
    scores = assemble_scores([rec], {"sub": _record_battles()}, "abc")
    assert len(scores) == 0


def test_assemble_h2h_skips_panel_hash_mismatch():
    rec = _record()
    rec["panel_hash"] = "WRONG"
    bundle = assemble_bundle(
        PANEL_META, ANCHOR_RATINGS, CALIBRATION, ANCHOR_H2H,
        [rec], {"sub": _record_battles()},
    )
    assert "sub" not in bundle["head_to_head"]["models"]


def test_no_bradley_terry_import():
    import judgearena.leaderboard.assemble as a
    src = inspect.getsource(a)
    assert "fit_bradley_terry" not in src


def test_anchor_rows_get_ci_from_calibration():
    # CALIBRATION has a point for "m1" with judge_ci [0.0, 2.0]; ANCHOR_RATINGS overall has m1, m2.
    bundle = assemble_bundle(PANEL_META, ANCHOR_RATINGS, CALIBRATION, ANCHOR_H2H, [], {})
    by_model = {r["model"]: r for r in bundle["rows"]}
    assert by_model["m1"]["ci_low"] == 0.0
    assert by_model["m1"]["ci_high"] == 2.0
    # m2 has no calibration point -> CI stays blank (None)
    assert by_model["m2"]["ci_low"] is None


def test_anchor_rows_get_winrate_when_present():
    ratings = {**ANCHOR_RATINGS, "winrate_overall": {"m1": 0.7, "m2": 0.3}}
    bundle = assemble_bundle(PANEL_META, ratings, CALIBRATION, ANCHOR_H2H, [], {})
    by_model = {r["model"]: r for r in bundle["rows"]}
    assert by_model["m1"]["winrate"] == 0.7
    assert by_model["m2"]["winrate"] == 0.3


def test_anchor_winrate_absent_is_backward_compatible():
    # ANCHOR_RATINGS has no winrate_overall key -> anchor winrate is None, no crash
    bundle = assemble_bundle(PANEL_META, ANCHOR_RATINGS, CALIBRATION, ANCHOR_H2H, [], {})
    by_model = {r["model"]: r for r in bundle["rows"]}
    assert by_model["m1"]["winrate"] is None


def test_pref_to_win_a():
    from judgearena.leaderboard.assemble import pref_to_win_a

    assert pref_to_win_a(0.1) == 1.0
    assert pref_to_win_a(0.9) == 0.0
    assert pref_to_win_a(0.5) == 0.5
    assert pref_to_win_a(None) is None
    assert pref_to_win_a(float("nan")) is None


def test_latest_panel_version_numeric_aware():
    from judgearena.leaderboard.assemble import latest_panel_version

    assert latest_panel_version(["v1", "v2", "v10"]) == "v10"
    assert latest_panel_version(["panel-v1", "panel-v10", "panel-v2"]) == "panel-v10"
    assert latest_panel_version(["smoke-10lang-v1"]) == "smoke-10lang-v1"
