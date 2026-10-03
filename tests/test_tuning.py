"""Tests for judge configuration tuning."""

from __future__ import annotations

import json
import subprocess

import pandas as pd
import pytest
from pydantic import ValidationError

from judgearena.config import RunConfig, load_config
from judgearena.tasks.registry import get_packaged_task
from judgearena.tuning.runner import run_tune_judge
from judgearena.tuning.search_space import (
    axis_overrides,
    build_neps_space,
    decode_config,
)

_SEARCH_SPACE = {
    "judge.temperature": [0.0, 0.5, 1.0],
    "prompt": [
        {"judge.prompt_preset": "meta-eval-pair-score", "judge.swap_mode": "fixed"},
        {"judge.prompt_preset": "meta-eval-pair-score", "judge.swap_mode": "both"},
    ],
}


def _tuning_config(tmp_path, **tune_judge):
    return RunConfig(
        task="tune-judge-comparia-fr",
        judge={"model": "Dummy/judge", "temperature": 0.0},
        run={"result_folder": str(tmp_path), "no_log_file": True},
        tune_judge={
            "search_space": _SEARCH_SPACE,
            "price_per_million_tokens": {"Dummy/judge": 1.0},
            "max_evaluations": 6,
            "min_battles_per_model": 2,
            "max_battles_per_model": 6,
            **tune_judge,
        },
    )


def test_neps_space_uses_base_config_as_prior(tmp_path):
    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path)
    base = cfg.model_dump(mode="json")
    base["judge"]["prompt_preset"] = "meta-eval-pair-score"
    space = build_neps_space(cfg.tune_judge, base)

    temperature, prompt = (
        space.searchables[k] for k in ("judge.temperature", "prompt")
    )
    assert json.loads(temperature.prior) == {"judge.temperature": 0.0}
    assert json.loads(prompt.prior) == _SEARCH_SPACE["prompt"][0]
    overrides, battles = decode_config(
        {
            "judge.temperature": temperature.choices[2],
            "prompt": prompt.choices[1],
            "battles_per_model": 6,
        }
    )
    assert overrides == {"judge.temperature": 1.0, **_SEARCH_SPACE["prompt"][1]}
    assert battles == 6
    with pytest.raises(ValueError, match="tuner-owned"):
        list(
            axis_overrides(
                _tuning_config(tmp_path).tune_judge.model_copy(
                    update={"search_space": {"meta_eval.split": ["test"]}}
                )
            )
        )


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
    pd.DataFrame(
        {"battle_id": ["b1"], "judge_input": ["prompt"], "judge_completion": ["x"]}
    ).to_parquet(run_dir / "annotations.parquet")


@pytest.mark.parametrize("algorithm", ["priorband", "mo_hyperband"])
def test_run_tune_judge_searches_validation_and_scores_pick_on_test(
    tmp_path, algorithm
):
    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path, algorithm=algorithm)
    test_results = run_tune_judge(
        cfg, get_packaged_task(cfg.task), execute_trial=_fake_trial
    )

    (tune_dir,) = tmp_path.glob("tune-*")
    trials = pd.read_parquet(tune_dir / "trials.parquet")
    assert len(trials) == 6
    assert set(trials["battles_per_model"]) <= {2, 6}
    (pick,) = test_results.itertuples()
    pick_cfg = load_config(tune_dir / "test" / pick.config_id / "config.yaml")
    assert (pick_cfg.task, pick_cfg.meta_eval.split) == (
        "meta-eval-comparia-fr",
        "test",
    )
    for trial_dir in (tune_dir / "trials").iterdir():
        trial_cfg = load_config(trial_dir / "config.yaml")
        assert trial_cfg.task == "meta-eval-comparia-fr"
        assert trial_cfg.meta_eval.split == "validation"
        assert trial_cfg.tune_judge is None


def test_tune_judge_requires_prices_and_tune_task(tmp_path):
    cfg = _tuning_config(tmp_path, price_per_million_tokens={})
    with pytest.raises(ValueError, match="missing models"):
        run_tune_judge(cfg, get_packaged_task(cfg.task), execute_trial=_fake_trial)
    with pytest.raises(ValidationError, match="tune-judge tasks"):
        RunConfig(
            task="meta-eval-comparia-fr",
            judge={"model": "Dummy/judge"},
            tune_judge=cfg.tune_judge.model_dump(),
        )
    with pytest.raises(ValidationError, match="tune-judge tasks"):
        RunConfig(task="tune-judge-comparia-fr", judge={"model": "Dummy/judge"})
