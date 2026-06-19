"""Tests for leaderboard publishing (offline; HF upload mocked)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from judgearena.leaderboard import publish
from judgearena.leaderboard.panel import PANEL_BATTLE_COLUMNS, Panel, save_panel
from judgearena.leaderboard.publish import build_bundle, build_scores_frame


def _panel():
    battles = pd.DataFrame(
        {
            "battle_id": ["b1", "b2", "b3", "b4"],
            "lang": ["en", "en", "fr", "fr"],
            "model_a": ["strong", "strong", "strong", "strong"],
            "model_b": ["weak", "weak", "weak", "weak"],
            "instruction": ["q1", "q2", "q3", "q4"],
            "completion_a": ["a", "a", "a", "a"],
            "completion_b": ["b", "b", "b", "b"],
            "human_winner": ["model_a", "model_a", "model_a", "model_a"],
            "judge_pref": [0.0, 0.0, 0.0, 0.0],
            "judge_pref_hard": [0.0, 0.0, 0.0, 0.0],
            "challenger_opponent": ["strong", "strong", "strong", "strong"],
            "challenger_position": ["A", "A", "A", "A"],
        }
    )
    meta = {
        "panel_version": "pv1", "panel_hash": "H", "baseline_model": "weak",
        "judge_model": "OpenRouter/judge", "mae_vs_human": 12.3,
        "kappa_per_language": {"en": 1.0, "fr": 1.0},
        "scorer": {"method": "soft", "temperature": 0.3, "calibrated": False},
    }
    return Panel(meta=meta, battles=battles[list(PANEL_BATTLE_COLUMNS)])


def _record(model, elo, tag=None, panel_hash="H"):
    return {
        "model": model, "tag": tag, "panel_hash": panel_hash,
        "elo_overall": elo, "elo_ci": [elo - 10, elo + 10], "n_battles": 4,
        "winrate_overall": 0.55, "elo_per_lang": {"en": elo + 1, "fr": elo - 1},
    }


def test_build_bundle_shape_and_meta():
    panel = _panel()
    records = [_record("cand", 1050.0, tag="seed-1"),
               _record("other-panel", 9999.0, panel_hash="X")]
    bundle = build_bundle(panel, records)

    assert bundle["panel"]["panel_version"] == "pv1"
    assert bundle["panel"]["judge_model"] == "OpenRouter/judge"
    assert bundle["panel"]["mae_vs_human"] == 12.3
    assert set(bundle["languages"]) == {"en", "fr"}

    models = [r["model"] for r in bundle["rows"]]
    assert "strong" in models and "weak" in models       # anchors
    assert "cand #seed-1" in models                       # tagged submission
    assert "other-panel" not in models                    # panel_hash mismatch dropped

    cand = next(r for r in bundle["rows"] if r["model"] == "cand #seed-1")
    assert cand["is_submission"] is True
    assert cand["winrate"] == 0.55
    assert cand["ci_low"] == 1040.0

    strong = next(r for r in bundle["rows"] if r["model"] == "strong")
    assert strong["is_submission"] is False
    assert strong["winrate"] is None
    assert strong["ci_low"] is None                       # NaN -> None

    # per-language ladders include anchors + the submission
    assert set(bundle["by_language"]) == {"en", "fr"}
    en_models = [r["model"] for r in bundle["by_language"]["en"]]
    assert "strong" in en_models and "cand #seed-1" in en_models


def test_build_scores_frame_longform():
    battles = pd.DataFrame({"lang": ["en", "fr"], "judge_pref": [0.2, 0.8]})
    rec = {"model": "cand", "tag": "seed-1"}
    df = build_scores_frame([(rec, battles)])
    assert list(df.columns) == ["model", "tag", "lang", "judge_pref"]
    assert len(df) == 2
    assert set(df["model"]) == {"cand"}
    assert set(df["lang"]) == {"en", "fr"}


def test_build_scores_frame_empty():
    df = build_scores_frame([])
    assert list(df.columns) == ["model", "tag", "lang", "judge_pref"]
    assert len(df) == 0


def _write_submission(results_dir, panel_version, rec, battles):
    d = results_dir / panel_version / rec["model"].replace("/", "_")
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(rec))
    battles.to_parquet(d / "battles.parquet", index=False)


def test_main_publish_writes_bundle_and_opens_pr(tmp_path, monkeypatch):
    panel = _panel()
    panel_dir = save_panel(panel, tmp_path / "panels" / "pv1")
    # Get the computed panel_hash from the saved panel
    saved_panel_meta = json.loads((Path(panel_dir) / "panel.json").read_text())
    panel_hash_computed = saved_panel_meta["panel_hash"]

    results = tmp_path / "results"
    _write_submission(
        results, "pv1", _record("cand", 1050.0, panel_hash=panel_hash_computed),
        pd.DataFrame({"lang": ["en", "fr"], "judge_pref": [0.3, 0.7]}),
    )

    captured = {}

    def fake_upload(repo, bundle_dir, *, token, create_pr):
        captured["repo"] = repo
        captured["create_pr"] = create_pr
        captured["files"] = sorted(p.name for p in Path(bundle_dir).iterdir())
        return "https://hf.co/datasets/x/pr/1"

    monkeypatch.setattr(publish, "_upload_bundle", fake_upload)

    out = publish.main_publish([
        "--panel-dir", str(panel_dir), "--repo", "user/leaderboard",
        "--results-dir", str(results), "--out", str(tmp_path / "bundle"),
    ])

    assert (out / "leaderboard.json").exists()
    assert (out / "scores.parquet").exists()
    bundle = json.loads((out / "leaderboard.json").read_text())
    assert any(r["model"] == "cand" for r in bundle["rows"])
    assert captured["repo"] == "user/leaderboard"
    assert captured["create_pr"] is True                  # PR by default
    assert set(captured["files"]) == {"leaderboard.json", "scores.parquet"}


def test_main_publish_push_commits_directly(tmp_path, monkeypatch):
    panel = _panel()
    panel_dir = save_panel(panel, tmp_path / "panels" / "pv1")
    # Get the computed panel_hash from the saved panel
    saved_panel_meta = json.loads((Path(panel_dir) / "panel.json").read_text())
    panel_hash_computed = saved_panel_meta["panel_hash"]

    results = tmp_path / "results"
    _write_submission(
        results, "pv1", _record("cand", 1050.0, panel_hash=panel_hash_computed),
        pd.DataFrame({"lang": ["en"], "judge_pref": [0.3]}),
    )
    seen = {}
    monkeypatch.setattr(publish, "_upload_bundle",
                        lambda repo, bundle_dir, *, token, create_pr: seen.setdefault("pr", create_pr) or "url")
    publish.main_publish([
        "--panel-dir", str(panel_dir), "--repo", "user/lb",
        "--results-dir", str(results), "--out", str(tmp_path / "b2"), "--push",
    ])
    assert seen["pr"] is False
