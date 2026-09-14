import io
import json
from contextlib import redirect_stdout

import pandas as pd

from judgearena.reports import BattleReport, EloReport
from judgearena.utils.eval import PrefSummary, compute_pref_summary


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


def test_report_compatibility_exports():
    from judgearena.benchmarks.elo.runner import EloReport as RunnerEloReport
    from judgearena.utils.eval import BattleReport as UtilsBattleReport

    assert RunnerEloReport is EloReport
    assert UtilsBattleReport is BattleReport


def test_battlereport_to_dict_arena_shape():
    report = BattleReport(
        task="alpaca-eval",
        model_a="my-model",
        model_b="gpt4",
        judge_model="judge",
        metrics={"pairwise_win_rate": _summary().to_dict()},
        swap_mode="fixed",
        result_folder="/tmp/run",
        preferences=[0.0, 1.0, 0.5, None],
        metadata={"baseline_assignment": "flat", "prompt_preset": "default"},
    )
    d = report.to_dict()
    assert d["schema_version"] == "2"
    assert d["report_type"] == "BattleReport"
    assert d["task"] == "alpaca-eval"
    assert d["model_A"] == "my-model"
    assert d["model_B"] == "gpt4"
    assert d["judge_model"] == "judge"
    assert d["swap_mode"] == "fixed"
    assert d["result_folder"] == "/tmp/run"
    assert d["metadata"]["baseline_assignment"] == "flat"
    assert d["metadata"]["prompt_preset"] == "default"
    assert d["metrics"]["pairwise_win_rate"]["num_wins"] == 2
    assert d["preferences"] == [0.0, 1.0, 0.5, None]
    assert "winrate" not in d
    assert "per_category" not in d
    assert "per_turn" not in d


def test_battlereport_to_dict_mtbench_shape():
    report = BattleReport(
        task="mt-bench",
        model_a="my-model",
        model_b="baseline",
        judge_model="judge",
        metrics={
            "pairwise_win_rate": {
                **_summary().to_dict(),
                "groups": {
                    "category": [
                        {
                            "group": "writing",
                            "values": _summary(
                                winrate=0.6, num_wins=3, num_losses=2, num_ties=0
                            ).to_dict(),
                        }
                    ],
                    "turn": [
                        {
                            "group": 1,
                            "values": _summary(
                                winrate=0.5, num_wins=1, num_losses=1, num_ties=0
                            ).to_dict(),
                        }
                    ],
                },
            }
        },
        preferences=[0.0, 1.0],
        metadata={"date": "2026-06-16", "user": "tester"},
    )
    d = report.to_dict()
    assert d["schema_version"] == "2"
    assert d["report_type"] == "BattleReport"
    assert d["model_A"] == "my-model"
    assert d["model_B"] == "baseline"
    assert d["metrics"] == report.metrics
    assert (
        d["metrics"]["pairwise_win_rate"]["groups"]["category"][0]["group"] == "writing"
    )
    assert d["metrics"]["pairwise_win_rate"]["groups"]["turn"][0]["group"] == 1
    assert (
        d["metrics"]["pairwise_win_rate"]["groups"]["category"][0]["values"]["winrate"]
        == 0.6
    )
    assert (
        d["metrics"]["pairwise_win_rate"]["groups"]["turn"][0]["values"]["winrate"]
        == 0.5
    )
    assert d["metadata"]["date"] == "2026-06-16"
    assert "per_category" not in d
    assert "per_turn" not in d
    assert "swap_mode" not in d
    assert "result_folder" not in d


def test_battlereport_render_arena_swap_both():
    report = BattleReport(
        task="alpaca-eval",
        model_a="A",
        model_b="B",
        judge_model="J",
        metrics={"pairwise_win_rate": _summary(num_battles=4, winrate=0.5).to_dict()},
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
    assert "pairwise_win_rate: 50.00%" in out
    assert "/tmp/x" in out


def test_battlereport_render_mtbench_breakdowns():
    report = BattleReport(
        task="mt-bench",
        model_a="A",
        model_b="B",
        judge_model="J",
        metrics={
            "pairwise_win_rate": {
                **_summary().to_dict(),
                "groups": {
                    "category": [
                        {
                            "group": "writing",
                            "values": _summary(
                                winrate=0.6, num_wins=3, num_losses=2, num_ties=0
                            ).to_dict(),
                        }
                    ],
                    "turn": [
                        {
                            "group": 1,
                            "values": _summary(
                                winrate=0.5, num_wins=1, num_losses=1, num_ties=0
                            ).to_dict(),
                        }
                    ],
                },
            }
        },
        preferences=[],
        metadata={},
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        report.render()
    out = buf.getvalue()
    assert "category=writing:" in out
    assert "writing" in out
    assert "turn=1:" in out


def test_battle_report_renders_metrics_and_groups(capsys):
    report = BattleReport(
        task="demo",
        model_a="candidate",
        model_b="baseline",
        judge_model="judge",
        metrics={
            "length_controlled_winrate": {
                "winrate": 0.52,
                "num_scored": 10,
                "num_pairs": 10,
                "groups": {
                    "category": [
                        {
                            "group": "writing",
                            "values": {
                                "winrate": 0.6,
                                "num_scored": 5,
                                "num_pairs": 5,
                            },
                        }
                    ]
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
        metrics={"pairwise_win_rate": _summary().to_dict()},
        swap_mode="fixed",
        result_folder="/tmp/run",
        preferences=[0.0, 1.0, 0.5],
        metadata={"baseline_assignment": "flat"},
    )

    path = report.save(tmp_path / "r.json")
    assert path.exists()
    loaded = json.loads(path.read_text())

    assert loaded == report.to_dict()
    assert loaded["schema_version"] == "2"
    assert loaded["report_type"] == "BattleReport"


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
    assert result == {
        "arena": "chatbot-arena",
        "judge_model": "judge",
        "metrics": {"bradley_terry": {"ratings": {"my-model": 1000.0}}},
        "num_battles": 10,
        "model_name": "my-model",
        "sampling_metadata": {"sampling_mode": "head"},
        "schema_version": "2",
        "report_type": "EloReport",
    }
