import pandas as pd
from judgearena.leaderboard.panel import Panel
from judgearena.leaderboard.pool_fit import extend_pool


def _panel_completions():
    battles = pd.DataFrame({
        "battle_id": ["b1"], "lang": ["en"], "model_a": ["m1"], "model_b": ["m2"],
        "instruction": ["q"], "completion_a": ["a1"], "completion_b": ["a2"],
        "human_winner": ["model_a"], "judge_pref": [0.2], "judge_pref_hard": [0.0],
        "challenger_opponent": [None], "challenger_position": [None],
    })
    meta = {"panel_version": "v1", "baseline_model": "m1",
            "pool_models": ["m1", "m2"], "anchor_models": ["m1", "m2"],
            "scorer": {"method": "soft", "temperature": 0.3}, "kappa_per_language": {"en": 0.7}}
    comps = pd.DataFrame({"model": ["m1", "m2"], "instruction": ["q", "q"],
                          "lang": ["en", "en"], "completion": ["a1", "a2"]})
    return Panel(meta=meta, battles=battles), comps


def test_extend_pool_adds_model_and_bumps_version():
    panel, comps = _panel_completions()
    new_battles = pd.DataFrame({
        "model_a": ["new", "new"], "model_b": ["m1", "m2"], "instruction": ["q", "q"],
        "lang": ["en", "en"], "judge_pref": [0.3, 0.4], "judge_pref_hard": [0.0, 0.0],
        "position": ["A", "A"], "opponent": ["m1", "m2"],
    })
    new_comps = pd.DataFrame({"model": ["new"], "instruction": ["q"], "lang": ["en"], "completion": ["nq"]})
    p2, c2 = extend_pool(panel, comps, "new", new_comps, new_battles, bump_version="v2")
    assert p2.meta["panel_version"] == "v2"
    assert "new" in p2.meta["pool_models"]
    assert p2.meta["anchor_models"] == ["m1", "m2"]      # unchanged: new is judge-only
    assert p2.meta["kappa_per_language"] == {"en": 0.7}  # carried over
    assert "new" in set(c2["model"])                     # completions merged
    assert len(p2.battles) == 3                          # 1 anchor + 2 new


def test_place_against_pool_returns_record_without_mutating_pool():
    from judgearena.leaderboard.pool_fit import place_against_pool
    panel, comps = _panel_completions()
    new_battles = pd.DataFrame({
        "model_a": ["new", "new"], "model_b": ["m1", "m2"], "instruction": ["q", "q"],
        "lang": ["en", "en"], "judge_pref": [0.3, 0.4], "judge_pref_hard": [0.0, 0.0],
        "position": ["A", "A"], "opponent": ["m1", "m2"],
    })
    before = len(panel.battles)
    rec = place_against_pool(panel, "new", new_battles, n_bootstraps=5, seed=0)
    assert rec.model == "new"
    assert isinstance(rec.elo_overall, float)
    assert len(panel.battles) == before  # pool not mutated
    # Regression: battles artifact must be populated so h2h/scores tabs see the record.
    assert len(rec.battles) == len(new_battles)
    assert {"opponent", "position", "judge_pref", "lang"}.issubset(rec.battles.columns)
