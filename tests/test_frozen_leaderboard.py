"""Focused regressions for fixed-reference scoring and execution."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_leaderboard_cli import setup_path as setup_path

import judgearena.models as models
from judgearena.benchmarks.elo.leaderboard import AnchorSet
from judgearena.benchmarks.elo.rating import fit_against_frozen_ratings
from judgearena.benchmarks.elo.runner import run_elo
from judgearena.benchmarks.elo.scoring import FrozenBradleyTerryResult
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.cli import cli
from judgearena.config import load_config
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import MetricSpec


def _anchors():
    return AnchorSet(
        name="euro-test",
        task="elo-test",
        arena="test-arena",
        baseline_model="reference",
        protocol_id="frozen-protocol",
        languages=("en", "fr"),
        ratings_by_language={
            "en": {"reference": 1000.0, "strong": 1200.0},
            "fr": {"reference": 1000.0, "strong": 1100.0},
        },
        counts_by_language={
            "en": {"reference": 20, "strong": 10},
            "fr": {"reference": 30, "strong": 15},
        },
        human_battles_by_language={"en": 40, "fr": 50},
        battles_per_language=4,
        bootstrap_seed=19,
    )


def test_fixed_bradley_terry_keeps_opponent_ratings_fixed():
    anchors = {"weaker": 800.0, "stronger": 1300.0}
    expected = 1100.0
    win_rates = {
        model: 1 / (1 + 10 ** ((rating - expected) / 400))
        for model, rating in anchors.items()
    }
    battles = pd.DataFrame(
        {
            "model_a": ["candidate", "stronger"],
            "model_b": ["weaker", "candidate"],
            "pref": [1 - win_rates["weaker"], win_rates["stronger"]],
        }
    )
    rating = fit_against_frozen_ratings(battles, "candidate", anchors)

    assert rating == pytest.approx(expected)
    assert anchors == {"weaker": 800.0, "stronger": 1300.0}


def test_fixed_rating_respects_bounds():
    battles = pd.DataFrame(
        {"model_a": ["candidate"], "model_b": ["reference"], "pref": [0.0]}
    )
    assert (
        fit_against_frozen_ratings(
            battles, "candidate", {"reference": 1000.0}, rating_bounds=(-10000, 10000)
        )
        == 10000
    )
    assert (
        fit_against_frozen_ratings(
            battles.assign(pref=1.0),
            "candidate",
            {"reference": 1000.0},
            rating_bounds=(-10000, 10000),
        )
        == -10000
    )


def test_metric_pipeline_rewards_better_judgments():
    anchors = _anchors()
    original = anchors.model_copy(deep=True)
    battles = pd.DataFrame(
        {
            "panel_id": ["en-0", "fr-0"],
            "lang": ["en", "fr"],
            "model_a": "candidate",
            "model_b": "reference",
            "evaluation_model": "candidate",
        }
    )
    metrics = build_metrics(
        (
            MetricSpec(metric="bradley_terry"),
            MetricSpec(metric="pairwise_win_rate"),
        )
    )
    estimates = []
    win_rates = []
    for preference in (0.8, 0.2):
        result = calculate_metrics(
            battles.assign(pref=preference),
            metrics,
            runtime_by_metric={"bradley_terry": {"anchors": anchors}},
        )
        estimates.append(
            FrozenBradleyTerryResult.model_validate(result["bradley_terry"])
        )
        win_rates.append(result["pairwise_win_rate"]["winrate"])

    assert estimates[1].entry.overall.rating > estimates[0].entry.overall.rating
    assert win_rates[1] > win_rates[0]
    assert anchors == original


def test_frozen_run_reuses_cache_and_saves_benchmark_seed(
    setup_path, tmp_path, monkeypatch
):
    board = tmp_path / "benchmark"
    cli(["leaderboard", "create", str(setup_path), "--output", str(board)])
    anchors = AnchorSet.load(board / "anchors.json")
    entries = []

    def no_new_inference(*_args, **_kwargs):
        raise AssertionError("A cached run must not construct a model backend")

    for index, seed in enumerate((11, 29)):
        directory = tmp_path / f"run-{index}"
        shutil.copytree(board, directory)
        cfg = load_config(directory / "config.yaml")
        cfg.model.name = "Dummy/candidate"
        cfg.elo.leaderboard_dir = directory
        cfg.run.result_folder = str(tmp_path / f"results-{index}")
        cfg.run.seed = seed
        if index:
            monkeypatch.setattr(models, "make_model", no_new_inference)
        result = run_elo(cfg, get_packaged_task(cfg.task))
        output = Path(result["result_path"]).parent
        entries.append(json.loads((output / "entry.json").read_text()))
        assert (
            json.loads((output / "elo_ratings.json").read_text())["seed"]
            == anchors.bootstrap_seed
        )

    assert entries[0] == entries[1]


def test_hard_preferences_are_combined_after_rounding_each_pass(
    setup_path, tmp_path, monkeypatch
):
    setup = yaml.safe_load(setup_path.read_text())
    setup["evaluation"]["elo"]["soft_elo"] = False
    setup_path.write_text(yaml.safe_dump(setup))
    board = tmp_path / "benchmark"
    cli(["leaderboard", "create", str(setup_path), "--output", str(board)])
    positions = pd.read_parquet(board / "panel.parquet")["candidate_position"]
    judge_pass = 0

    def batch(self, inputs, **_kwargs):
        nonlocal judge_pass
        if self.name == "Dummy/candidate":
            return ["candidate response"] * len(inputs)
        judge_pass += 1
        # The candidate loses strongly in direct order, then wins narrowly
        # in reverse order, regardless of its assigned panel position.
        scores = (
            {"A": "score A: 0 score B: 10", "B": "score A: 10 score B: 0"}
            if judge_pass == 1
            else {"A": "score A: 4 score B: 5", "B": "score A: 5 score B: 4"}
        )
        return [scores[position] for position in positions]

    monkeypatch.setattr(models.DummyModel, "batch", batch)
    cli(["leaderboard", "evaluate", str(board), "--model", "Dummy/candidate"])
    entry = json.loads(next((board / "entries").glob("*.json")).read_text())
    # Each pair is one hard win and one hard loss. Averaging soft scores before
    # rounding instead would incorrectly produce a decisive preference.
    assert entry["overall"]["rating"] == pytest.approx(1000)
