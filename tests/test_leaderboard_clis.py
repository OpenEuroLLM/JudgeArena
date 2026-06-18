"""Tests for the leaderboard CLIs (offline; judge/model/generation mocked)."""

from __future__ import annotations

import json

import pandas as pd

from judgearena.estimate_elo_ratings import _slugify
from judgearena.leaderboard.panel import PANEL_BATTLE_COLUMNS, Panel, save_panel
from judgearena.leaderboard.record import ResultRecord


def _save_tiny_panel(directory):
    meta = {
        "panel_version": "pv1",
        "judge_model": "OpenRouter/judge",
        "baseline_model": None,
        "generation_params": {"max_out_tokens": 256, "truncate_all_input_chars": 1000},
        "scorer": {"method": "soft", "temperature": 0.3, "calibrated": False},
    }
    battles = pd.DataFrame(columns=list(PANEL_BATTLE_COLUMNS))
    return save_panel(Panel(meta=meta, battles=battles), directory)


def test_main_submit_uses_panel_judge_and_writes_record(tmp_path, monkeypatch):
    import judgearena.leaderboard.submit as sub

    panel_dir = _save_tiny_panel(tmp_path / "panel")
    captured = {}

    monkeypatch.setattr(sub, "generate_panel_completions", lambda panel, model, **kw: ["c1", "c2"])

    def fake_score(panel, completions, *, model, judge_cfg, n_bootstraps, seed,
                   generation_params, scorer=None):
        captured["judge"] = judge_cfg.model
        captured["gen"] = generation_params
        captured["n_bootstraps"] = n_bootstraps
        return ResultRecord(
            model=model, panel_version=panel.meta["panel_version"],
            panel_hash=panel.meta["panel_hash"], judge_model=judge_cfg.model,
            elo_overall=1010.0, elo_std=5.0, elo_ci=[1000.0, 1020.0],
            elo_per_lang={}, winrate_overall=0.5, winrate_per_lang={},
            n_battles=2, n_battles_per_lang={}, kappa_per_lang={},
            mae_vs_human=float("nan"), scorer=panel.meta["scorer"],
            generation_params=generation_params, seed=seed,
        )

    monkeypatch.setattr(sub, "score_against_panel", fake_score)

    model = "VLLM/openeurollm/OLMo-3-7B-Dolci-Translated-A-75EN"
    out = sub.main_submit([
        "--panel-dir", str(panel_dir), "--model", model,
        "--out", str(tmp_path / "results"), "--n-bootstraps", "7",
    ])

    assert out == tmp_path / "results" / "pv1" / _slugify(model)
    assert (out / "result.json").exists()
    payload = json.loads((out / "result.json").read_text())
    assert payload["elo_overall"] == 1010.0
    assert captured["judge"] == "OpenRouter/judge"        # panel's frozen judge
    assert captured["gen"]["max_out_tokens"] == 256       # panel's frozen generation
    assert captured["n_bootstraps"] == 7
