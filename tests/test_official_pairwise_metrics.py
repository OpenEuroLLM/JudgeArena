"""Official benchmark metric regressions."""

import hashlib
from pathlib import Path

import numpy as np
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
from judgearena.benchmarks.scoring import (
    build_metric,
    build_metrics,
    calculate_metrics,
    render_metrics,
)
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import MetricSpec


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


def test_arena_hard_protocols_handle_an_incomplete_order_pair_differently():
    battles = _battles(
        [0.0, 0.25, None, 0.75],
        instruction_index=["q0", "q1", "q0", "q1"],
        orientation=["direct", "direct", "reversed", "reversed"],
        category="creative_writing",
    )

    v01 = ArenaHardV01Metric().calculate(battles)
    v20 = ArenaHardV20Metric().calculate(battles)

    # Historical v0.1 retains each parseable judgment independently. V2 drops
    # both orientations when either judgment in one judge's pair is missing.
    assert v01["num_missing"] == 1
    assert v01["winrate"] == pytest.approx(0.8)
    assert v20["num_missing"] == 2
    assert v20["winrate"] == pytest.approx(0.495)


def test_arena_hard_v2_uses_official_category_methods(monkeypatch):
    monkeypatch.setattr(arena_hard, "BOOTSTRAP_ROUNDS", 3)
    hard_id = "0001b527ced3428d"
    battles = _battles(
        [0.0, 1.0, 0.0, 1.0],
        instruction_index=[hard_id, "creative", hard_id, "creative"],
        baseline=[
            "o3-mini-2025-01-31",
            "gemini-2.0-flash-001",
            "o3-mini-2025-01-31",
            "gemini-2.0-flash-001",
        ],
        judge="gpt-4.1",
        judge_prompt_preset="arena-hard",
        judge_temperature=0.0,
        judge_max_out_tokens=16000,
        orientation=["direct", "direct", "reversed", "reversed"],
        category=["hard_prompt", "creative_writing"] * 2,
    )
    requests = (MetricSpec(metric="arena_hard_v20", group_by=("category",)),)

    result = calculate_metrics(battles, build_metrics(requests))["arena_hard_v20"]
    groups = {item["group"]: item["values"] for item in result["groups"]["category"]}

    assert groups["hard_prompt"]["scoring_method"] == "joint_style_controlled_bt"
    assert groups["hard_prompt"]["raw_winrate"] == 1.0
    assert groups["creative_writing"]["scoring_method"] == "weighted_mean"
    assert groups["creative_writing"]["winrate"] == 0.0
    assert result["aggregate_score_is_official"] is False
    assert "unofficial aggregate" in render_metrics({"arena_hard_v20": result})


def test_arena_hard_v2_pools_complete_pairs_per_judge():
    battles = _battles(
        [0.0, 1.0, 0.0, 1.0],
        instruction_index=["creative"] * 4,
        orientation=["direct", "reversed", "direct", "reversed"],
        judge=["gpt-4.1", "gpt-4.1", "gemini-2.5", "gemini-2.5"],
        category="creative_writing",
    )

    result = ArenaHardV20Metric().calculate(battles)

    assert result["num_missing"] == 0
    assert (result["num_wins"], result["num_losses"]) == (2, 2)
    assert 0.0 <= result["winrate"] <= 1.0


def test_arena_hard_v2_calibration_artifact_is_pinned():
    artifact = Path(arena_hard.__file__).with_name("arena_hard_v20_calibration.csv.gz")
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "83fa4e19343e119faa80fb7aaefda4916c218b071cbbccdd083c54ddda458d50"
    )
    population = arena_hard._load_style_calibration()
    assert population.shape == (53_978, 15)
    assert population["model"].nunique() == 27
    assert population["instruction_index"].nunique() == 500


def test_arena_hard_v2_reproduces_published_full_population_score():
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


@pytest.mark.parametrize(
    ("judge", "prompt", "message"),
    [
        ("gpt-4.1", "custom", "requires judge_prompt_preset='arena-hard'"),
        ("other-judge", "arena-hard", "no calibration"),
    ],
)
def test_arena_hard_v2_rejects_unmatched_calibration(judge, prompt, message):
    battles = _battles(
        [0.5, 0.5],
        instruction_index=["0001b527ced3428d"] * 2,
        baseline="o3-mini-2025-01-31",
        judge=judge,
        judge_prompt_preset=prompt,
        judge_temperature=0.0,
        judge_max_out_tokens=16000,
        orientation=["direct", "reversed"],
        category="hard_prompt",
    )

    with pytest.raises(ValueError, match=message):
        ArenaHardV20Metric().calculate(battles)


def test_arena_hard_style_features_match_upstream_extraction():
    completion = """# Header
1. ordered
- unordered
**bold** and __also bold__
```
# ignored
- ignored
**ignored**
```
"""

    assert _style_features(completion).tolist() == [31.0, 1.0, 2.0, 2.0]


@pytest.mark.parametrize("preference", [2.0, float("inf")])
def test_official_pairwise_metrics_reject_invalid_preferences(preference):
    battles = _battles([0.5, preference])
    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        _alpaca_metric().calculate(battles)
    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        ArenaHardV01Metric().calculate(battles)


def test_alpaca_eval_metric_maps_rows_and_normalizes_percentages(monkeypatch):
    captured = {}

    def fake_score(annotations, **parameters):
        captured["annotations"] = annotations.copy()
        captured["parameters"] = parameters
        return {
            "length_controlled_winrate": 75.0,
            "lc_standard_error": 1.0,
            "win_rate": 75.0,
        }

    monkeypatch.setattr(alpaca_scoring, "_length_controlled_metrics", fake_score)
    result = _alpaca_metric().calculate(_battles([0.25, None]))

    assert len(captured["annotations"]) == 2
    assert captured["parameters"] == {
        "calibration_repo_id": "tatsu-lab/alpaca_eval",
        "calibration_filename": "df_gamed.csv",
        "calibration_revision": "2edc6fad8be6b14ea7230aabfd08188da6b8b814",
        "gamed_weight": 0.1,
    }
    assert captured["annotations"]["preference"].iloc[0] == 1.75
    assert pd.isna(captured["annotations"]["preference"].iloc[1])
    assert result["num_missing"] == 1
    assert result["length_controlled_winrate"] == 0.75
    assert result["lc_standard_error"] == 0.01
    assert result["raw_winrate"] == 0.75


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


def test_alpaca_eval_lc_matches_pinned_reference_value(monkeypatch):
    download = alpaca_scoring.hf_hub_download

    def local_only_download(*args, **kwargs):
        kwargs["local_files_only"] = True
        try:
            return download(*args, **kwargs)
        except OSError:
            pytest.skip("pinned AlpacaEval calibration is not cached")

    monkeypatch.setattr(alpaca_scoring, "hf_hub_download", local_only_download)
    rng = np.random.default_rng(0)
    n = 805
    battles = _battles(
        rng.uniform(0, 1, n).tolist(),
        completion_model=["x" * int(value) for value in rng.integers(50, 2000, n)],
        completion_baseline=["y" * int(value) for value in rng.integers(50, 2000, n)],
    )

    result = _alpaca_metric().calculate(battles)
    with_missing = battles.copy()
    with_missing.loc[:49, "pref"] = np.nan
    missing_result = _alpaca_metric().calculate(with_missing)

    assert result["length_controlled_winrate"] == pytest.approx(0.4829068772368286)
    assert result["lc_standard_error"] == pytest.approx(0.0019811122623953162)
    assert result["raw_winrate"] == pytest.approx(0.4820877535856965)
    # Missing judge parses are excluded from training/raw rate, but upstream
    # still predicts them in the fixed 805-instruction LC evaluation population.
    assert missing_result["length_controlled_winrate"] == pytest.approx(
        0.481848434267128
    )
    assert missing_result["lc_standard_error"] == pytest.approx(0.0019634150580809208)
    assert missing_result["raw_winrate"] == pytest.approx(0.48267870759097653)
