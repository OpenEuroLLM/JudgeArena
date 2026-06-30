import io
import json
from contextlib import redirect_stdout

import pandas as pd

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
    assert d["schema_version"] == "1"
    assert d["report_type"] == "BattleReport"
    assert d["task"] == "alpaca-eval"
    assert d["model_A"] == "my-model"
    assert d["model_B"] == "gpt4"
    assert d["judge_model"] == "judge"
    assert d["swap_mode"] == "fixed"
    assert d["result_folder"] == "/tmp/run"
    assert d["metadata"]["baseline_assignment"] == "flat"
    assert d["metadata"]["prompt_preset"] == "default"
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
    assert d["schema_version"] == "1"
    assert d["report_type"] == "BattleReport"
    assert d["per_category"]["writing"]["winrate"] == 0.6
    assert d["per_turn"][1]["winrate"] == 0.5
    assert d["metadata"]["date"] == "2026-06-16"
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


def test_battlereport_save_round_trip(tmp_path):
    report = BattleReport(
        task="alpaca-eval",
        model_a="my-model",
        model_b="gpt4",
        judge_model="judge",
        summary=_summary(),
        swap_mode="fixed",
        result_folder="/tmp/run",
        preferences=[0.0, 1.0, 0.5],
        metadata={"baseline_assignment": "flat"},
    )
    path = report.save(tmp_path / "r.json")
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded == report.to_dict()
    assert loaded["schema_version"] == "1"
    assert loaded["report_type"] == "BattleReport"


def test_eloreport_to_dict_envelope():
    from judgearena.estimate_elo_ratings import EloReport

    report = EloReport(
        arena="chatbot-arena",
        judge_model="judge",
        summary=_summary(),
        num_battles=10,
        llm_judged_battles=10,
        human_anchor_battles=5,
        elo_mean=1000.0,
        elo_std=10.0,
        elo_num_bootstraps=100,
        mae_vs_human=5.0,
        method="Soft-ELO",
        n_bootstraps=100,
        model_name="my-model",
        mean_ratings={"my-model": 1000.0},
        battle_counts={"my-model": 10},
        human_elo={"gpt4": 1100.0},
        bootstrap_ratings=[{"my-model": 1000.0}],
        sampling_metadata={"sampling_mode": "head"},
        source_battle_counts={"my-model": 10},
    )
    d = report.to_dict()
    assert d["schema_version"] == "1"
    assert d["report_type"] == "EloReport"
    assert d["arena"] == "chatbot-arena"
    assert d["model_A"] == "my-model"
