"""Tests for the leaderboard CLIs (offline; judge/model/generation mocked)."""

from __future__ import annotations

import json

import pandas as pd

from judgearena.estimate_elo_ratings import _slugify
from judgearena.leaderboard.board import build_board, main_board, render_board
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


def _board_panel():
    battles = pd.DataFrame(
        {
            "battle_id": ["b1", "b2", "b3"],
            "lang": ["en", "en", "fr"],
            "model_a": ["strong", "strong", "strong"],
            "model_b": ["weak", "weak", "weak"],
            "instruction": ["q1", "q2", "q3"],
            "completion_a": ["a", "a", "a"],
            "completion_b": ["b", "b", "b"],
            "human_winner": ["model_a", "model_a", "model_a"],
            "judge_pref": [0.0, 0.0, 0.0],          # strong (A) always wins
            "judge_pref_hard": [0.0, 0.0, 0.0],
            "challenger_opponent": ["strong", "strong", "strong"],
            "challenger_position": ["A", "A", "A"],
        }
    )
    meta = {
        "panel_version": "pv1", "panel_hash": "H", "baseline_model": "weak",
        "scorer": {"method": "soft", "temperature": 0.3, "calibrated": False},
    }
    return Panel(meta=meta, battles=battles)


def _record(model, elo, panel_hash="H"):
    return {
        "model": model, "panel_hash": panel_hash, "elo_overall": elo,
        "elo_ci": [elo - 10, elo + 10], "n_battles": 3,
        "elo_per_lang": {"en": elo + 1, "fr": elo - 1},
    }


def test_build_board_merges_anchors_and_submissions_sorted():
    panel = _board_panel()
    records = [_record("cand-mid", 1050.0), _record("cand-other-panel", 9999.0, panel_hash="X")]
    board = build_board(panel, records)
    # anchors present, matching submission present, mismatched submission excluded
    models = list(board["model"])
    assert "strong" in models and "weak" in models and "cand-mid" in models
    assert "cand-other-panel" not in models
    # sorted by elo descending with ranks 1..n
    assert list(board["rank"]) == list(range(1, len(board) + 1))
    assert board["elo"].is_monotonic_decreasing
    # the submission flag is set only on the submitted model
    assert bool(board.loc[board["model"] == "cand-mid", "is_submission"].iloc[0]) is True
    assert bool(board.loc[board["model"] == "strong", "is_submission"].iloc[0]) is False


def test_build_board_lang_uses_per_language_values():
    panel = _board_panel()
    board = build_board(panel, [_record("cand-mid", 1050.0)], lang="en")
    assert board.loc[board["model"] == "cand-mid", "elo"].iloc[0] == 1051.0  # elo_per_lang["en"]


def test_render_board_formats():
    panel = _board_panel()
    board = build_board(panel, [_record("cand-mid", 1050.0)])
    assert "cand-mid" in render_board(board, "table")
    assert "| Rank |" in render_board(board, "markdown")
    assert "rank,model,elo" in render_board(board, "csv").splitlines()[0]


def test_main_board_reads_records_and_prints(tmp_path, capsys):
    # a panel on disk (real panel_hash) + one matching record under results/<pv>/<slug>/
    panel_dir = _save_tiny_panel(tmp_path / "panel")  # empty-battles panel is fine here
    from judgearena.leaderboard.panel import load_panel
    ph = load_panel(panel_dir).meta["panel_hash"]

    rec_dir = tmp_path / "results" / "pv1" / "cand-mid"
    rec_dir.mkdir(parents=True)
    (rec_dir / "result.json").write_text(json.dumps({
        "model": "cand-mid", "panel_hash": ph, "elo_overall": 1050.0,
        "elo_ci": [1040.0, 1060.0], "n_battles": 3, "elo_per_lang": {},
    }))

    main_board([
        "--panel-dir", str(panel_dir),
        "--results-dir", str(tmp_path / "results"),
        "--format", "markdown",
    ])
    out = capsys.readouterr().out
    assert "cand-mid" in out
    assert "| Rank |" in out


def test_main_submit_tag_suffixes_dir_and_sets_record(tmp_path, monkeypatch):
    import judgearena.leaderboard.submit as sub
    from judgearena.estimate_elo_ratings import _slugify

    panel_dir = _save_tiny_panel(tmp_path / "panel")
    monkeypatch.setattr(sub, "generate_panel_completions", lambda panel, model, **kw: ["c"])

    saved = {}

    def fake_score(panel, completions, *, model, judge_cfg, n_bootstraps, seed,
                   generation_params, scorer=None):
        from judgearena.leaderboard.record import ResultRecord
        rec = ResultRecord(
            model=model, panel_version=panel.meta["panel_version"],
            panel_hash=panel.meta["panel_hash"], judge_model=judge_cfg.model,
            elo_overall=1000.0, elo_std=0.0, elo_ci=[1000.0, 1000.0],
            elo_per_lang={}, winrate_overall=0.5, winrate_per_lang={},
            n_battles=1, n_battles_per_lang={}, kappa_per_lang={},
            mae_vs_human=float("nan"), scorer={}, generation_params=generation_params,
            seed=seed,
        )
        saved["rec"] = rec
        return rec

    monkeypatch.setattr(sub, "score_against_panel", fake_score)

    model = "OpenRouter/deepseek/deepseek-v3.2"
    out = sub.main_submit([
        "--panel-dir", str(panel_dir), "--model", model,
        "--out", str(tmp_path / "results"), "--tag", "seed-1",
    ])
    assert out.name == f"{_slugify(model)}__{_slugify('seed-1')}"
    assert (out / "result.json").exists()
    assert saved["rec"].tag == "seed-1"


def test_build_board_distinguishes_same_model_by_tag():
    from judgearena.leaderboard.board import build_board
    panel = _board_panel()
    ph = panel.meta["panel_hash"]
    rec_a = {"model": "deepseek", "tag": "seed-1", "panel_hash": ph,
             "elo_overall": 1010.0, "elo_ci": [1000.0, 1020.0], "n_battles": 3, "elo_per_lang": {}}
    rec_b = {"model": "deepseek", "tag": "seed-2", "panel_hash": ph,
             "elo_overall": 1005.0, "elo_ci": [995.0, 1015.0], "n_battles": 3, "elo_per_lang": {}}
    board = build_board(panel, [rec_a, rec_b])
    labels = list(board["model"])
    assert "deepseek #seed-1" in labels
    assert "deepseek #seed-2" in labels
