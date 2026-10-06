"""Contracts for native NePS judge tuning and persisted trial artifacts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from judgearena.config import RunConfig, dump_config, load_config
from judgearena.tasks.registry import get_packaged_task
from judgearena.tuning import runner
from judgearena.tuning.search_space import (
    FIDELITY,
    apply_overrides,
    axis_overrides,
    build_neps_space,
    decode_config,
)


def _tuning_config(tmp_path, algorithm="neps_priorband", **settings):
    optimizer = {"name": algorithm, "eta": 3}
    objectives = ["agreement"]
    if algorithm in {"mo_hyperband", "primo"}:
        objectives.append("cost_per_1k_battles")
    if algorithm == "primo":
        optimizer.update(
            initial_design_size=2,
            prior_confidences={
                f"objective_{i}": {"judge.temperature": 0.25, "judge.model": 0.25}
                for i in range(2)
            },
            prior_centers={
                f"objective_{i}": {
                    "judge.temperature": value,
                    "judge.model": "Dummy/judge",
                }
                for i, value in enumerate((0.2, 0.8))
            },
        )
    specs = {
        "judge.temperature": dict(type="Float", lower=0.0, upper=1.0),
        "judge.model": dict(type="Categorical", choices=["Dummy/judge", "Dummy/other"]),
        FIDELITY: dict(type="IntegerFidelity", lower=1, upper=3),
    }
    for key, value in [("judge.temperature", 0.2), ("judge.model", 1)]:
        specs[key].update(prior=value, prior_confidence="medium")
    return RunConfig(
        task="tune-judge-comparia-fr",
        judge={"model": "Dummy/judge", "temperature": 0.2, "swap_mode": "both"},
        run={"result_folder": str(tmp_path), "no_log_file": True},
        tune_judge={
            "neps": {
                "pipeline_space": specs,
                "optimizer": optimizer,
                "total_evaluations_to_spend": 10,
            },
            "objectives": objectives,
            "price_per_million_tokens": {"Dummy/judge": 1.0, "Dummy/other": 1.0},
            "run_dir": tmp_path / "tuning",
            **settings,
        },
    )


@pytest.fixture
def token_counter(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_judge_token_encoding",
        lambda: SimpleNamespace(encode=lambda text, **kwargs: text.split()),
    )


def _fake_trial(config_path):
    cfg = load_config(config_path)
    folder = config_path.parent / "run"
    folder.mkdir()
    agreement = 0.6 - 0.1 * cfg.judge.temperature
    agreement_metrics = dict(
        accuracy_attempted=agreement,
        cohen_kappa=agreement,
        coverage=1.0,
        n_attempted=cfg.meta_eval.battles_per_model,
    )
    metrics = {"meta_eval_agreement": {"all": agreement_metrics}}
    (folder / "results.json").write_text(json.dumps({"metrics": metrics}))
    pd.DataFrame(
        {
            "battle_id": ["b1", "b1"],
            "judge_input": ["system forward", "system reverse"],
            "judge_completion": ["answer", "answer"],
            "cache_hit": [True, True],
        }
    ).to_parquet(folder / "annotations.parquet")


@pytest.mark.parametrize(
    "algorithm", ["neps_priorband", "neps_hyperband", "mo_hyperband", "primo"]
)
def test_native_optimizer_smoke(tmp_path, token_counter, algorithm):
    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path, algorithm)
    calls = []

    def execute(path):
        calls.append(path)
        _fake_trial(path)

    task = get_packaged_task(cfg.task)
    results = runner.run_tune_judge(cfg, task, execute_trial=execute)
    assert not results.empty and set(results.status) == {"completed"}
    if algorithm != "neps_priorband":
        return
    folder = cfg.tune_judge.run_dir
    trials = pd.read_parquet(folder / "trials.parquet")
    assert trials.battles_per_model.max() == 3
    assert trials.neps_trial_id.is_unique
    best = trials[trials.battles_per_model == 3].groupby("judge_model").agreement.max()
    assert results.set_index("judge_model").agreement.to_dict() == best.to_dict()
    for path in calls:
        trial = load_config(path)
        assert trial.task == "meta-eval-comparia-fr" and trial.tune_judge is None
        assert trial.meta_eval.split == (
            "test" if "test" in path.parts else "validation"
        )
    assert load_config(calls[-1]).meta_eval.battles_per_model == 3
    before = len(calls)
    (folder / "test_results.parquet").unlink()
    resumed = runner.run_tune_judge(cfg, task, execute_trial=execute)
    pd.testing.assert_frame_equal(results, resumed)
    assert len(calls) == before  # Completed search and held-out artifacts are reused.
    helper = _tuning_config(tmp_path, search_only=True)
    assert runner.run_tune_judge(helper, task, execute_trial=execute).empty
    assert len(calls) == before
    incompatible = _tuning_config(
        tmp_path, price_per_million_tokens={"Dummy/judge": 2.0, "Dummy/other": 2.0}
    )
    with pytest.raises(ValueError, match="Incompatible configuration"):
        runner.run_tune_judge(incompatible, task, execute_trial=execute)
    assert len(calls) == before
    fresh = _tuning_config(tmp_path, run_dir=tmp_path / "fresh")
    runner.run_tune_judge(fresh, task, execute_trial=execute)
    repeated = pd.read_parquet(fresh.tune_judge.run_dir / "trials.parquet")
    assert repeated.overrides.tolist() == trials.overrides.tolist()
    extended = _tuning_config(tmp_path)
    extended.tune_judge.neps["total_evaluations_to_spend"] = 11
    runner.run_tune_judge(extended, task, execute_trial=execute)
    assert len(pd.read_parquet(folder / "trials.parquet")) == 11
    runner.run_tune_judge(helper, task, execute_trial=execute)
    from neps.state import NePSState

    assert (
        NePSState.create_or_load(
            folder / "neps", load_only=True
        ).lock_and_get_global_budgets()[0]
        == 11
    )


def test_dotted_domains_and_index_priors(tmp_path):
    neps = pytest.importorskip("neps")
    specs = _tuning_config(tmp_path).tune_judge.neps["pipeline_space"]
    specs["judge.max_out_tokens"] = dict(type="Integer", lower=16, upper=32)
    attrs = build_neps_space(specs).get_attrs()
    assert attrs["judge.model"].prior == 1
    assert attrs["judge.model"].choices == ("Dummy/judge", "Dummy/other")
    assert isinstance(attrs["judge.temperature"], neps.Float)
    assert attrs["judge.temperature"].prior == 0.2
    assert isinstance(attrs["judge.max_out_tokens"], neps.Integer)
    assert isinstance(attrs[FIDELITY], neps.IntegerFidelity)
    assert {"judge.temperature": 1.0} in list(axis_overrides(specs))
    overrides, battles = decode_config(
        {"judge.temperature": 0.37, "judge.model": "Dummy/other", FIDELITY: 3}
    )
    base = {"judge": {"temperature": 0.2}}
    assert apply_overrides(base, overrides)["judge"] == {
        "temperature": 0.37,
        "model": "Dummy/other",
    }
    assert base["judge"]["temperature"] == 0.2 and battles == 3
    for name, spec in [
        ("meta_eval.split", {"type": "Categorical", "choices": ["test"]}),
        ("judge.temperature", {"type": "IntegerFidelity", "lower": 1, "upper": 3}),
    ]:
        with pytest.raises(ValueError, match="tuner-owned|Only"):
            build_neps_space({**specs, name: spec})


def test_budget_end_does_not_wait_for_unstarted_batch(
    tmp_path, token_counter, monkeypatch
):
    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path)
    cfg.tune_judge.neps.update(
        optimizer={"name": "neps_random_search", "ignore_fidelity": "highest_fidelity"},
        total_evaluations_to_spend=2,
        sample_batch_size=3,
    )
    monkeypatch.setattr(
        "judgearena.tuning.session.time.sleep",
        lambda _: pytest.fail("Waiting for an unstarted trial"),
    )
    results = runner.run_tune_judge(
        cfg, get_packaged_task(cfg.task), execute_trial=_fake_trial
    )
    assert not results.empty
    assert len(pd.read_parquet(cfg.tune_judge.run_dir / "trials.parquet")) == 2


def test_cost_counts_both_cached_orientations(tmp_path, token_counter):
    path = tmp_path / "config.yaml"
    dump_config(_tuning_config(tmp_path), path)
    _fake_trial(path)
    record = runner._read_trial(tmp_path, price_per_million_tokens=2.0)
    assert record["tokens_per_battle"] == 6
    assert record["cost_per_1k_battles"] == pytest.approx(0.012)
    objective = runner._objective(
        {**record, "status": "completed"}, ["agreement", "cost_per_1k_battles"]
    )
    assert objective["objective_to_minimize"] == pytest.approx([0.42, 0.012])
    assert "cost" not in objective


@pytest.mark.parametrize(
    "change,match",
    [
        ({"neps": {"total_cost_to_spend": 1}}, "managed"),
        ({"neps": {"root_directory": "elsewhere"}}, "managed"),
        ({"neps": {}}, "finite global budget"),
        ({"search_only": True, "run_dir": None}, "explicit"),
        ({"neps": {"worker_evaluations_to_spend": 1}}, "Worker-local"),
        ({"objectives": ["agreement", "cost_per_1k_battles"]}, "single objective"),
        ({"neps": {"optimizer": {"name": "mo_hyperband"}}}, "agreement and cost"),
        ({"price_per_million_tokens": {"Dummy/judge": -1}}, "nonnegative"),
    ],
)
def test_tuning_config_validation(tmp_path, change, match):
    values = _tuning_config(tmp_path).model_dump()
    if "neps" in change and change["neps"]:
        change = {**change, "neps": {**values["tune_judge"]["neps"], **change["neps"]}}
    values["tune_judge"].update(change)
    with pytest.raises(ValidationError, match=match):
        RunConfig(**values)


def test_prices_required_before_execution(tmp_path):
    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path, price_per_million_tokens={})
    with pytest.raises(ValueError, match="missing models"):
        runner.run_tune_judge(
            cfg,
            get_packaged_task(cfg.task),
            execute_trial=lambda _: pytest.fail("executed without prices"),
        )
