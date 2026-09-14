import json

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


def _summary():
    return PrefSummary(
        num_battles=4, winrate=0.5, num_wins=2, num_losses=1, num_ties=1, num_missing=0
    )


def test_battlereport_save_and_render(tmp_path, capsys):
    report = BattleReport(
        task="mt-bench",
        model_a="A",
        model_b="B",
        judge_model="judge",
        summary=_summary(),
        swap_mode="both",
        result_folder="/tmp/run",
        preferences=[0.0, 1.0, 0.5],
        metadata={"prompt_preset": "default"},
        per_category={
            "writing": {"winrate": 0.6, "num_wins": 3, "num_losses": 2, "num_ties": 0}
        },
        per_turn={1: {"winrate": 0.5, "num_wins": 1, "num_losses": 1, "num_ties": 0}},
    )
    loaded = json.loads(report.save(tmp_path / "r.json").read_text())
    assert loaded["schema_version"] == "1"
    assert loaded["report_type"] == "BattleReport"
    assert loaded["num_wins"] == 2
    assert loaded["preferences"] == [0.0, 1.0, 0.5]
    assert loaded["metadata"]["prompt_preset"] == "default"
    assert loaded["per_category"]["writing"]["winrate"] == 0.6
    assert loaded["per_turn"]["1"]["winrate"] == 0.5

    report.render()
    out = capsys.readouterr().out
    assert "Win Rate (A): 50.0%" in out
    assert "both orders" in out
    assert "/tmp/run" in out
    assert "Per-Category Breakdown:" in out
    assert "writing" in out
    assert "Per-Turn Breakdown:" in out


def test_eloreport_to_dict_envelope():
    from judgearena.benchmarks.elo.runner import EloReport

    report = EloReport(
        arena="chatbot-arena",
        judge_model="judge",
        summary=_summary(),
        num_battles=10,
        llm_judged_battles=10,
        human_anchor_battles=5,
        elo_mean=1000.0,
        elo_std=10.0,
        elo_n_bootstraps=100,
        mae_vs_human=5.0,
        method="Soft-ELO",
        n_bootstraps=100,
        model_name="my-model",
        mean_ratings={"my-model": 1000.0},
        battle_counts={"my-model": 10},
        human_elo={"gpt4": 1100.0},
        bootstrap_ratings=[{"my-model": 1000.0}],
        sampling_metadata={"sampling_mode": "head"},
    )
    d = report.to_dict()
    assert d["schema_version"] == "1"
    assert d["report_type"] == "EloReport"
    assert d["arena"] == "chatbot-arena"
    assert d["model_name"] == "my-model"
