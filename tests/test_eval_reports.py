import json

import pandas as pd

from judgearena.utils.eval import BattleReport, PrefSummary, compute_pref_summary


def test_compute_pref_summary_returns_win_loss_tie_rate():
    prefs = pd.Series([0.0, 0.2, 1.0, 0.5, None])
    summary = compute_pref_summary(prefs)

    assert isinstance(summary, PrefSummary)
    assert summary.num_battles == 5
    assert summary.num_wins == 2
    assert summary.num_losses == 1
    assert summary.num_ties == 1
    assert summary.num_missing == 1
    assert summary.winrate == (2 + 0.5) / 4


def _summary(
    num_battles=4, winrate=0.5, num_wins=2, num_losses=1, num_ties=1, num_missing=0
):
    return PrefSummary(
        num_battles=num_battles,
        winrate=winrate,
        num_wins=num_wins,
        num_losses=num_losses,
        num_ties=num_ties,
        num_missing=num_missing,
    )


def test_battle_report_serializes_metrics_as_the_result():
    metrics = {
        "pairwise_win_rate": {
            "winrate": 0.5,
            "num_battles": 4,
            "groups": {"category": [{"group": "writing", "values": {"winrate": 0.6}}]},
        }
    }
    report = BattleReport(
        task="mt-bench",
        model_a="my-model",
        model_b="baseline",
        judge_model="judge",
        metrics=metrics,
        preferences=[0.0, 1.0],
        metadata={"date": "2026-06-16", "user": "tester"},
    )

    result = report.to_dict()

    assert result["schema_version"] == "2"
    assert result["report_type"] == "BattleReport"
    assert result["metrics"] == metrics
    assert result["model_A"] == "my-model"
    assert result["model_B"] == "baseline"
    assert "winrate" not in result
    assert "per_category" not in result
    assert "per_turn" not in result


def test_battle_report_renders_metrics_and_groups(capsys):
    report = BattleReport(
        task="demo",
        model_a="candidate",
        model_b="baseline",
        judge_model="judge",
        metrics={
            "length_controlled_winrate": {
                "winrate": 0.52,
                "groups": {
                    "category": [{"group": "writing", "values": {"winrate": 0.6}}]
                },
            }
        },
        result_folder="/tmp/run",
    )

    report.render()
    output = capsys.readouterr().out

    assert "length_controlled_winrate" in output
    assert "category=writing" in output
    assert "/tmp/run" in output


def test_battle_report_save_round_trip(tmp_path):
    report = BattleReport(
        task="alpaca-eval",
        model_a="my-model",
        model_b="gpt4",
        judge_model="judge",
        metrics={"pairwise_win_rate": {"winrate": 0.5}},
        swap_mode="fixed",
        result_folder="/tmp/run",
        preferences=[0.0, 1.0, 0.5],
        metadata={"baseline_assignment": "flat"},
    )

    path = report.save(tmp_path / "r.json")
    loaded = json.loads(path.read_text())

    assert loaded == report.to_dict()
    assert loaded["schema_version"] == "2"


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
    result = report.to_dict()
    assert result["schema_version"] == "1"
    assert result["report_type"] == "EloReport"
    assert result["arena"] == "chatbot-arena"
    assert result["model_name"] == "my-model"
