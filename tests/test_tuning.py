"""Tests for judge configuration tuning."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from judgearena.config import RunConfig, load_config
from judgearena.tuning.runner import run_tune_judge
from judgearena.tuning.search_space import apply_overrides, expand_grid
from judgearena.tuning.selection import select_survivors


def test_expand_grid_combines_plain_and_grouped_axes():
    trials = expand_grid(
        {
            "judge.temperature": [0.0, 1.0],
            "prompt": [
                {"judge.prompt_preset": "meta-eval-pair-score"},
                {"judge.prompt_preset": "alpaca-eval", "judge.top_logprobs": 20},
            ],
        }
    )

    assert len({trial.id for trial in trials}) == 4
    assert {
        "judge.temperature": 1.0,
        "judge.prompt_preset": "alpaca-eval",
        "judge.top_logprobs": 20,
    } in [trial.overrides for trial in trials]
    assert apply_overrides({"judge": {"model": "m"}}, trials[0].overrides)["judge"] == {
        "model": "m",
        "temperature": 0.0,
        "prompt_preset": "meta-eval-pair-score",
    }
    with pytest.raises(ValueError, match="tuner-owned"):
        expand_grid({"meta_eval.split": ["test"]})


def test_select_survivors_sorts_pareto_fronts_by_agreement_first():
    agreement = np.array([0.50, 0.60, 0.55, 0.40])
    cost = np.array([1.0, 2.0, 3.0, 0.5])

    assert select_survivors(agreement, cost, n_keep=2, min_agreement=0.45) == [1, 0]
    assert select_survivors(agreement, cost, n_keep=3, min_agreement=0.45) == [1, 0, 2]


def _fake_trial(config_path):
    cfg = load_config(config_path)
    if cfg.judge.temperature == 1.0 and cfg.judge.swap_mode == "both":
        raise subprocess.CalledProcessError(1, "judgearena")
    run_dir = config_path.parent / "run"
    run_dir.mkdir()
    agreement = 0.6 - 0.1 * cfg.judge.temperature
    metrics = {
        "meta_eval_agreement": {
            "all": {
                "accuracy_attempted": agreement,
                "cohen_kappa": agreement,
                "coverage": 1.0,
                "n_attempted": cfg.meta_eval.battles_per_model,
            }
        }
    }
    (run_dir / "results.json").write_text(json.dumps({"metrics": metrics}))
    completion = "x " * (1 + 10 * (cfg.judge.swap_mode == "both"))
    pd.DataFrame(
        {
            "battle_id": ["b1"],
            "judge_input": ["prompt"],
            "judge_completion": [completion],
        }
    ).to_parquet(run_dir / "annotations.parquet")


def _tuning_config(tmp_path, **tune_judge):
    return RunConfig(
        task="meta-eval-comparia",
        judge={"model": "Dummy/judge"},
        run={"result_folder": str(tmp_path), "no_log_file": True},
        tune_judge={
            "search_space": {
                "judge.temperature": [0.0, 0.5, 1.0],
                "judge.swap_mode": ["fixed", "both"],
            },
            "rungs": [2, 4],
            "price_per_million_tokens": {"Dummy/judge": 1.0},
            **tune_judge,
        },
    )


def test_run_tune_judge_halves_on_validation_and_scores_pick_on_test(tmp_path):
    test_results = run_tune_judge(_tuning_config(tmp_path), execute_trial=_fake_trial)

    (tune_dir,) = tmp_path.glob("tune-*")
    trials = pd.read_parquet(tune_dir / "trials.parquet")
    assert trials.groupby("rung").size().tolist() == [6, 2]
    assert (trials["status"] == "failed").sum() == 1
    (pick,) = test_results["trial_id"]
    pick_cfg = load_config(tune_dir / "test" / pick / "config.yaml")
    assert pick_cfg.meta_eval.split == "test"
    assert (pick_cfg.judge.temperature, pick_cfg.judge.swap_mode) == (0.0, "fixed")
    for rung in trials.itertuples():
        rung_cfg = load_config(
            tune_dir / f"rung-{rung.rung}" / rung.trial_id / "config.yaml"
        )
        assert rung_cfg.meta_eval.split == "validation"
        assert rung_cfg.tune_judge is None


def test_tune_judge_requires_prices_and_meta_eval_task(tmp_path):
    with pytest.raises(ValueError, match="missing models"):
        run_tune_judge(
            _tuning_config(tmp_path, price_per_million_tokens={}),
            execute_trial=_fake_trial,
        )
    with pytest.raises(ValidationError, match="only valid for meta-evaluation"):
        RunConfig(
            task="alpaca-eval",
            model={"name": "Dummy/a"},
            judge={"model": "Dummy/judge"},
            tune_judge={
                "search_space": {"judge.temperature": [0.0]},
                "price_per_million_tokens": {},
            },
        )
