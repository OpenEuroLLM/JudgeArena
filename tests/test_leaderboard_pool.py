import pandas as pd
from judgearena.leaderboard.panel import Panel
from judgearena.leaderboard.pool import (
    POOL_COMPLETION_COLUMNS, save_pool, load_pool, pool_completion,
    pool_models, anchor_models, completions_from_battles,
)


def _panel_and_completions(tmp):
    battles = pd.DataFrame({
        "battle_id": ["b1"], "lang": ["en"], "model_a": ["m1"], "model_b": ["m2"],
        "instruction": ["q"], "completion_a": ["a1"], "completion_b": ["a2"],
        "human_winner": ["model_a"], "judge_pref": [0.2], "judge_pref_hard": [0.0],
        "challenger_opponent": [None], "challenger_position": [None],
    })
    meta = {"panel_version": "v1", "baseline_model": "m1",
            "pool_models": ["m1", "m2"], "anchor_models": ["m1", "m2"]}
    comps = pd.DataFrame({
        "model": ["m1", "m2"], "instruction": ["q", "q"], "lang": ["en", "en"],
        "completion": ["a1", "a2"],
    })
    return Panel(meta=meta, battles=battles), comps


def _make_battles(*rows):
    """Build a minimal battles DataFrame from (model_a, model_b, instruction, completion_a, completion_b, lang) tuples."""
    return pd.DataFrame(rows, columns=["model_a", "model_b", "instruction", "completion_a", "completion_b", "lang"])


def test_completions_from_battles_basic():
    battles = _make_battles(("m1", "m2", "q", "a1", "a2", "en"))
    out = completions_from_battles(battles)
    assert list(out.columns) == POOL_COMPLETION_COLUMNS
    assert len(out) == 2
    assert pool_completion(out, "m1", "q") == "a1"
    assert pool_completion(out, "m2", "q") == "a2"


def test_completions_from_battles_deduplication():
    # Two battle rows sharing the same (model, instruction) pair → deduplicated to one.
    battles = _make_battles(
        ("m1", "m2", "q", "a1", "a2", "en"),
        ("m1", "m3", "q", "a1", "a3", "en"),
    )
    out = completions_from_battles(battles)
    assert len(out[out["model"] == "m1"]) == 1


def test_completions_from_battles_empty():
    battles = pd.DataFrame(
        columns=["model_a", "model_b", "instruction", "completion_a", "completion_b", "lang"]
    )
    out = completions_from_battles(battles)
    assert list(out.columns) == POOL_COMPLETION_COLUMNS
    assert len(out) == 0


def test_save_load_pool_roundtrip(tmp_path):
    panel, comps = _panel_and_completions(tmp_path)
    save_pool(panel, comps, tmp_path / "v1")
    p2, c2 = load_pool(tmp_path / "v1")
    assert list(c2.columns) == POOL_COMPLETION_COLUMNS
    assert pool_models(p2) == ["m1", "m2"]
    assert anchor_models(p2) == ["m1", "m2"]
    assert pool_completion(c2, "m2", "q") == "a2"
    assert pool_completion(c2, "absent", "q") is None
