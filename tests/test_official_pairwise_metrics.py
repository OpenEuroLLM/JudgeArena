"""Official benchmark metric regressions."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from judgearena.benchmarks.pairwise.scoring import alpaca_eval as alpaca_scoring
from judgearena.benchmarks.pairwise.scoring import arena_hard
from judgearena.benchmarks.pairwise.scoring.alpaca_eval import (
    AlpacaEvalLengthControlledMetric,
)
from judgearena.benchmarks.pairwise.scoring.arena_hard import (
    ArenaHardV01Metric,
    ArenaHardV20Metric,
    _style_features,
)
from judgearena.benchmarks.scoring import build_metric
from judgearena.tasks.registry import get_packaged_task


def _alpaca_metric() -> AlpacaEvalLengthControlledMetric:
    request = get_packaged_task("alpaca-eval").spec.protocol.scoring.metrics[0]
    metric = build_metric(request.metric, request.parameters)
    assert isinstance(metric, AlpacaEvalLengthControlledMetric)
    return metric


def _battles(prefs: list, **overrides) -> pd.DataFrame:
    n = len(prefs)
    columns = {
        "instruction_index": [f"{i:04d}" for i in range(n)],
        "model": "model-under-test",
        "baseline": "baseline-model",
        "completion_model": ["m" * (i + 1) for i in range(n)],
        "completion_baseline": ["b" * (2 * i + 1) for i in range(n)],
        "pref": pd.Series(prefs, dtype="float64"),
    }
    columns.update(overrides)
    return pd.DataFrame(columns)


def test_arena_hard_v01_weighting_and_bootstrap_match_official_fixture():
    metric = ArenaHardV01Metric()
    weighted = metric.calculate(_battles([0.0, 0.25, 0.75, None]))
    fixture = metric.calculate(
        _battles([0.0, 0.5, 0.75] * 10, category="arena-hard-v0.1")
    )

    assert weighted["winrate"] == pytest.approx(0.8)
    assert (weighted["num_wins"], weighted["num_losses"]) == (2, 1)
    assert weighted["num_missing"] == 1
    assert fixture["winrate"] == pytest.approx(0.7)
    assert fixture["score_ci_low"] == pytest.approx(0.6)
    assert fixture["score_ci_high"] == pytest.approx(0.8)


def test_arena_hard_protocols_handle_incomplete_pairs_differently():
    battles = _battles(
        [0.0, None, 0.25, 0.75, 0.0, 1.0],
        instruction_index=["q0", "q0", "q1", "q1", "q1", "q1"],
        orientation=["direct", "reversed"] * 3,
        judge=["judge-1"] * 4 + ["judge-2"] * 2,
        category="creative_writing",
    )

    v01 = ArenaHardV01Metric().calculate(battles)
    v20 = ArenaHardV20Metric().calculate(battles)

    assert v01["num_missing"] == 1  # v0.1 keeps each parseable judgment
    assert v20["num_missing"] == 2  # v2 drops the incomplete judge/order pair
    assert (v20["num_wins"], v20["num_losses"]) == (2, 2)


def test_arena_hard_v2_selects_official_method_per_category():
    result = ArenaHardV20Metric().calculate(
        _battles([0.0, 1.0], category=["hard_prompt", "creative_writing"])
    )

    assert result["category_methods"] == {
        "hard_prompt": "joint_style_controlled_bt",
        "creative_writing": "weighted_mean",
    }
    assert result["aggregate_score_is_official"] is False


def test_arena_hard_v2_rejects_uncalibrated_judge():
    with pytest.raises(ValueError, match="no calibration"):
        arena_hard._calibration_judge("other-judge")


def test_arena_hard_v2_matches_pinned_official_golden():
    artifact = Path(arena_hard.__file__).with_name("arena_hard_v20_calibration.csv.gz")
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "83fa4e19343e119faa80fb7aaefda4916c218b071cbbccdd083c54ddda458d50"
    )
    assert _style_features("# Header\n1. item\n- item\n**bold**").tolist() == [
        13.0,
        1.0,
        2.0,
        1.0,
    ]
    population = arena_hard._load_style_calibration()
    live = population.loc[
        (population["judge"] == "gpt-4.1") & (population["model"] == "deepseek-r1")
    ].copy()
    live["judge_prompt_preset"] = "arena-hard"
    live["judge_temperature"] = 0.0
    live["judge_max_out_tokens"] = 16000

    calibration, complete = arena_hard._select_calibration(live)
    result = ArenaHardV20Metric().calculate(live)

    assert complete is True
    assert "deepseek-r1" not in {
        arena_hard._fit_model_id(model) for model in calibration["model"]
    }
    assert result["official_population_complete"] is True
    assert result["winrate"] == pytest.approx(0.4854, abs=0.0002)
    assert result["score_ci_low"] < 0.48 < result["score_ci_high"]


def test_alpaca_eval_lc_synthetic_golden_runs_offline(monkeypatch):
    gamed = pd.DataFrame(
        [
            {
                "index": index,
                "preference": (index + baseline + 1) / 12,
                "std_delta_len": (index - 4.5) / (2 + baseline),
                "instruction_difficulty": (index - 4.5) / 5,
                "not_gamed_baseline": False,
            }
            for index in range(10)
            for baseline in range(2)
        ]
    )
    monkeypatch.setattr(alpaca_scoring, "_load_gamed_data", lambda *_args: gamed)
    battles = _battles(
        [0.1, 0.2, None, 0.4, 0.5, 0.6, 0.7, None, 0.9, 1.0],
        completion_model=["x" * (20 + index**2) for index in range(10)],
        completion_baseline=["y" * (12 + 3 * index) for index in range(10)],
    )

    result = _alpaca_metric().calculate(battles)

    assert result["length_controlled_winrate"] == pytest.approx(0.7497336730523078)
    assert result["lc_standard_error"] == pytest.approx(0.04840456960699923)
    assert result["raw_winrate"] == pytest.approx(0.45)
    assert result["num_missing"] == 2
