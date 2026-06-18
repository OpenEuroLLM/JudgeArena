import io
import json
from contextlib import redirect_stdout

import pandas as pd

from judgearena.estimate_elo_ratings import EloReport
from judgearena.utils.eval import BattleReport, PrefSummary, compute_pref_summary


def test_compute_pref_summary_returns_prefsummary():
    # 0.0 = A wins, 1.0 = B wins, 0.5 = tie, None = missing
    prefs = pd.Series([0.0, 0.0, 1.0, 0.5, None])
    summary = compute_pref_summary(prefs)
    assert isinstance(summary, PrefSummary)
    assert summary.num_battles == 5
    assert summary.num_wins == 2
    assert summary.num_losses == 1
    assert summary.num_ties == 1
    assert summary.num_missing == 1
    assert summary.winrate == (2 + 0.5 * 1) / 4


def test_prefsummary_to_dict_keys():
    prefs = pd.Series([0.0, 1.0])
    keys = set(compute_pref_summary(prefs).to_dict().keys())
    assert keys == {
        "num_battles",
        "winrate",
        "num_wins",
        "num_losses",
        "num_ties",
        "num_missing",
    }


def _summary(num_battles=4, winrate=0.5, num_wins=2, num_losses=1, num_ties=1, num_missing=0):
    return PrefSummary(
        num_battles=num_battles,
        winrate=winrate,
        num_wins=num_wins,
        num_losses=num_losses,
        num_ties=num_ties,
        num_missing=num_missing,
    )


def test_battlereport_to_dict_arena_shape():
    report = BattleReport(
        task="alpaca-eval",
        model_a="my-model",
        model_b="gpt4",
        judge_model="judge",
        summary=_summary(),
        swap_mode="fixed",
        result_folder="/tmp/run",
        preferences=[0.0, 1.0, 0.5, None],
        metadata={"baseline_assignment": "flat", "prompt_preset": "default"},
    )
    d = report.to_dict()
    assert d["task"] == "alpaca-eval"
    assert d["model_A"] == "my-model"
    assert d["model_B"] == "gpt4"
    assert d["judge_model"] == "judge"
    assert d["swap_mode"] == "fixed"
    assert d["result_folder"] == "/tmp/run"
    assert d["baseline_assignment"] == "flat"
    assert d["prompt_preset"] == "default"
    assert d["num_wins"] == 2
    assert d["preferences"] == [0.0, 1.0, 0.5, None]
    assert "per_category" not in d
    assert "per_turn" not in d


def test_battlereport_to_dict_mtbench_shape():
    report = BattleReport(
        task="mt-bench",
        model_a="my-model",
        model_b="baseline",
        judge_model="judge",
        summary=_summary(),
        per_category={"writing": {"winrate": 0.6, "num_wins": 3, "num_losses": 2, "num_ties": 0}},
        per_turn={1: {"winrate": 0.5, "num_wins": 1, "num_losses": 1, "num_ties": 0}},
        preferences=[0.0, 1.0],
        metadata={"date": "2026-06-16", "user": "tester"},
    )
    d = report.to_dict()
    assert d["per_category"]["writing"]["winrate"] == 0.6
    assert d["per_turn"][1]["winrate"] == 0.5
    assert d["date"] == "2026-06-16"
    assert "swap_mode" not in d
    assert "result_folder" not in d


def test_battlereport_render_arena_swap_both():
    report = BattleReport(
        task="alpaca-eval",
        model_a="A",
        model_b="B",
        judge_model="J",
        summary=_summary(num_battles=4, winrate=0.5),
        swap_mode="both",
        result_folder="/tmp/x",
        preferences=[],
        metadata={},
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        report.render()
    out = buf.getvalue()
    assert "MODEL BATTLE RESULTS" in out
    assert "Win Rate (A): 50.0%" in out
    assert "both orders" in out
    assert "/tmp/x" in out


def test_battlereport_render_mtbench_breakdowns():
    report = BattleReport(
        task="mt-bench",
        model_a="A",
        model_b="B",
        judge_model="J",
        summary=_summary(),
        per_category={"writing": {"winrate": 0.6, "num_wins": 3, "num_losses": 2, "num_ties": 0}},
        per_turn={1: {"winrate": 0.5, "num_wins": 1, "num_losses": 1, "num_ties": 0}},
        preferences=[],
        metadata={},
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        report.render()
    out = buf.getvalue()
    assert "Per-Category Breakdown:" in out
    assert "writing" in out
    assert "Per-Turn Breakdown:" in out


def _elo_report():
    return EloReport(
        arena="LMArena-100k",
        model_a="my/model",
        judge_model="judge",
        summary=_summary(num_battles=10, winrate=0.6, num_wins=6, num_losses=3, num_ties=1),
        num_battles=10,
        llm_judged_battles=10,
        human_anchor_battles=500,
        elo_mean=1050.0,
        elo_std=12.0,
        elo_num_bootstraps=100,
        mae_vs_human=8.0,
        method="Soft-ELO",
        calibrated_temperature=1.0,
        n_bootstraps=100,
        model_name="my/model",
        mean_ratings={"my/model": 1050.0, "gpt4": 1100.0},
        battle_counts={"my/model": 10, "gpt4": 8},
        human_elo={"gpt4": 1092.0},
        bootstrap_ratings=[{"my/model": 1050.0, "gpt4": 1100.0}],
        sampling_metadata={"strategy": "head"},
        source_battle_counts={"my/model": 10, "gpt4": 8},
    )


def test_eloreport_to_dict_shape():
    d = _elo_report().to_dict()
    assert d["arena"] == "LMArena-100k"
    assert d["model_A"] == "my/model"
    assert d["judge_model"] == "judge"
    assert d["num_battles"] == 10
    assert d["elo_mean"] == 1050.0
    assert d["llm_judged_battles"] == 10
    assert d["source_battle_counts"] == {"my/model": 10, "gpt4": 8}
    assert d["num_wins"] == 6  # PrefSummary core spread in


def test_eloreport_render_smoke():
    buf = io.StringIO()
    with redirect_stdout(buf):
        _elo_report().render()
    out = buf.getvalue()
    assert "Results for my/model" in out
    assert "Soft-ELO" in out
    assert "gpt4" in out
    assert "<-----" in out  # focal-model marker
    assert "MAE vs Human-ELO" in out


def test_eloreport_save_payload(tmp_path):
    path = _elo_report().save(tmp_path)
    payload = json.loads(path.read_text())
    assert set(payload.keys()) == {"summary", "bootstrap_ratings"}
    assert payload["summary"]["model_A"] == "my/model"
    assert payload["bootstrap_ratings"] == [{"my/model": 1050.0, "gpt4": 1100.0}]
