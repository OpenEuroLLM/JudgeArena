# tests/test_leaderboard_anchors.py
import pandas as pd
import pytest

from judgearena.leaderboard.anchors import (
    compute_anchor_h2h,
    compute_anchor_ratings,
    compute_calibration,
    save_anchor_caches,
)
from judgearena.leaderboard.assemble import load_anchor_caches
from judgearena.leaderboard.panel import Panel, panel_hash


def _panel() -> Panel:
    # Two langs, three anchor models, deterministic prefs (0 => model_a wins).
    rows = []
    for lang in ("en", "de"):
        for a, b, jp, hw in [
            ("m1", "m2", 0.1, "model_a"),
            ("m2", "m3", 0.2, "model_a"),
            ("m1", "m3", 0.15, "model_a"),
        ]:
            rows.append(
                {
                    "battle_id": f"{lang}-{a}-{b}",
                    "lang": lang,
                    "model_a": a,
                    "model_b": b,
                    "instruction": "q",
                    "completion_a": "x",
                    "completion_b": "y",
                    "human_winner": hw,
                    "judge_pref": jp,
                    "judge_pref_hard": 0.0,
                    "challenger_opponent": b,
                    "challenger_position": "A",
                }
            )
    battles = pd.DataFrame(rows)
    meta = {
        "panel_version": "v1",
        "panel_hash": panel_hash(battles),
        "baseline_model": "m1",
        "scorer": {"method": "soft", "temperature": 0.3},
        "kappa_per_language": {"en": 0.7, "de": 0.6},
    }
    return Panel(meta=meta, battles=battles)


def test_anchor_ratings_shape_and_counts():
    ratings = compute_anchor_ratings(_panel())
    assert set(ratings) == {"overall", "per_lang", "counts_overall", "counts_per_lang", "winrate_overall"}
    assert set(ratings["overall"]) == {"m1", "m2", "m3"}
    # each model appears in 2 battles per lang -> 4 overall
    assert ratings["counts_overall"]["m1"] == 4
    assert set(ratings["per_lang"]) == {"en", "de"}
    assert ratings["counts_per_lang"]["en"]["m1"] == 2


def test_calibration_points_and_mae():
    cal = compute_calibration(_panel(), n_bootstrap=5, seed=0)
    assert set(cal) == {"mae", "spearman", "points", "ci"}
    assert {p["model"] for p in cal["points"]} == {"m1", "m2", "m3"}
    assert all(len(p["judge_ci"]) == 2 for p in cal["points"])


def test_anchor_h2h_is_symmetric():
    h2h = compute_anchor_h2h(_panel())["pairwise"]
    # m1 vs m2 fought in 2 langs -> 2 battles each direction
    assert h2h["m1"]["m2"][1] == 2
    assert h2h["m2"]["m1"][1] == 2
    # judge_pref 0.1 < 0.5 => model_a (m1) wins both -> wins == 2.0
    assert h2h["m1"]["m2"][0] == pytest.approx(2.0)
    assert h2h["m2"]["m1"][0] == pytest.approx(0.0)


def test_anchor_winrate_overall():
    ratings = compute_anchor_ratings(_panel())
    wr = ratings["winrate_overall"]
    assert set(wr) == {"m1", "m2", "m3"}
    # In _panel(), every battle has judge_pref < 0.5 (model_a always wins).
    # m1 is model_a in all its battles -> wins all -> 1.0.
    assert wr["m1"] == pytest.approx(1.0)
    # m3 is model_b in all its battles -> loses all -> 0.0.
    assert wr["m3"] == pytest.approx(0.0)
    assert 0.0 <= wr["m2"] <= 1.0


def test_human_and_judge_elo_helpers():
    from judgearena.leaderboard.anchors import (
        human_elo_from_battles,
        judge_elo_from_battles,
    )

    panel = _panel()
    he = human_elo_from_battles(panel.battles, "m1")
    je = judge_elo_from_battles(panel.battles, "judge_pref", "m1")
    assert set(he) == {"m1", "m2", "m3"}
    assert set(je) == {"m1", "m2", "m3"}
    # m1 wins all its battles, m3 loses all -> m1 highest, m3 lowest (any centering)
    assert he["m1"] == max(he.values())
    assert he["m3"] == min(he.values())
    assert je["m1"] == max(je.values())
    assert je["m3"] == min(je.values())


def test_save_and_load_roundtrip(tmp_path):
    panel = _panel()
    save_anchor_caches(panel, tmp_path)
    for name in ("anchor_ratings.json", "calibration.json", "anchor_h2h.json"):
        assert (tmp_path / name).exists()
    ratings, cal, h2h = load_anchor_caches(tmp_path)
    assert ratings["overall"].keys() == compute_anchor_ratings(panel)["overall"].keys()
    assert "points" in cal
    assert "pairwise" in h2h


def test_calibration_ci_covers_all_boot_models():
    cal = compute_calibration(_panel(), n_bootstrap=5, seed=0)
    assert "ci" in cal
    ci = cal["ci"]
    # _panel() has m1, m2, m3 — all should appear in ci
    assert set(ci.keys()) == {"m1", "m2", "m3"}
    for model, bounds in ci.items():
        assert isinstance(bounds, list) and len(bounds) == 2, f"ci[{model!r}] is not a 2-element list"
        lo, hi = bounds
        assert lo <= hi, f"ci[{model!r}] lo > hi"
