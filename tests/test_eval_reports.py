import json

import pandas as pd

from judgearena.reports import BattleReport, EloReport
from judgearena.utils.eval import PrefSummary, compute_pref_summary


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


def test_report_compatibility_exports():
    from judgearena.benchmarks.elo.runner import EloReport as RunnerEloReport
    from judgearena.utils.eval import BattleReport as UtilsBattleReport

    assert RunnerEloReport is EloReport
    assert UtilsBattleReport is BattleReport


def test_battle_report_saves_and_renders_metrics(tmp_path, capsys):
    metrics = {
        "length_controlled_winrate": {
            "winrate": 0.52,
            "num_scored": 10,
            "num_pairs": 10,
            "groups": {
                "category": [
                    {
                        "group": "writing",
                        "values": {"winrate": 0.6, "num_scored": 5, "num_pairs": 5},
                    }
                ]
            },
        }
    }
    report = BattleReport(
        task="mt-bench",
        model_a="candidate",
        model_b="baseline",
        judge_model="judge",
        metrics=metrics,
        result_folder="/tmp/run",
        preferences=[0.0, 1.0, 0.5],
        metadata={"prompt_preset": "default"},
    )
    result = json.loads(report.save(tmp_path / "r.json").read_text())
    assert result["schema_version"] == "2"
    assert result["report_type"] == "BattleReport"
    assert result["metrics"] == metrics
    assert result["model_A"] == "candidate"
    assert result["preferences"] == [0.0, 1.0, 0.5]
    assert result["metadata"]["prompt_preset"] == "default"
    assert "winrate" not in result

    report.render()
    output = capsys.readouterr().out
    assert "length_controlled_winrate" in output
    assert "category=writing" in output
    assert "/tmp/run" in output


def test_eloreport_to_dict_envelope():
    report = EloReport(
        arena="chatbot-arena",
        judge_model="judge",
        metrics={"bradley_terry": {"ratings": {"my-model": 1000.0}}},
        num_battles=10,
        model_name="my-model",
        sampling_metadata={"sampling_mode": "head"},
    )

    result = report.to_dict()
    assert result["schema_version"] == "2"
    assert result["report_type"] == "EloReport"
    assert result["metrics"]["bradley_terry"]["ratings"] == {"my-model": 1000.0}
