"""Native NePS execution and persisted trial artifacts."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from judgearena.config import RunConfig, dump_config, load_config
from judgearena.tasks.registry import get_packaged_task
from judgearena.tuning import runner
from judgearena.tuning.search_space import FIDELITY


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
        task="tune-judge",
        judge={"model": "Dummy/judge", "temperature": 0.2, "swap_mode": "both"},
        run={"result_folder": str(tmp_path), "no_log_file": True},
        tune_judge={
            "meta_eval_task": "meta-eval-comparia-fr",
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
    changed_target = _tuning_config(tmp_path, meta_eval_task="meta-eval-comparia-en")
    with pytest.raises(ValueError, match="Incompatible configuration"):
        runner.run_tune_judge(changed_target, task, execute_trial=execute)
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


def test_prices_required_before_execution(tmp_path):
    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path, price_per_million_tokens={})
    with pytest.raises(ValueError, match="missing models"):
        runner.run_tune_judge(
            cfg,
            get_packaged_task(cfg.task),
            execute_trial=lambda _: pytest.fail("executed without prices"),
        )


@pytest.mark.parametrize("algorithm", ["neps_priorband", "mo_hyperband"])
def test_failed_trial_does_not_stop_search(tmp_path, token_counter, algorithm):
    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path, algorithm)
    cfg.tune_judge.neps["ignore_errors"] = True
    calls = []

    def execute(path):
        calls.append(path)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, ["judge-trial"])
        _fake_trial(path)

    results = runner.run_tune_judge(
        cfg, get_packaged_task(cfg.task), execute_trial=execute
    )
    trials = pd.read_parquet(cfg.tune_judge.run_dir / "trials.parquet")
    assert len(trials) == 10
    assert (trials.status == "failed").sum() == 1
    assert not results.empty and set(results.status) == {"completed"}
    pareto = pd.read_parquet(cfg.tune_judge.run_dir / "pareto.parquet")
    final = trials[(trials.status == "completed") & (trials.battles_per_model == 3)]
    assert set(pareto.agreement) == {final.agreement.max()}
    assert set(pareto.status) == {"completed"}
    assert set(pareto.battles_per_model) == {3}


def test_pareto_report_uses_completed_full_fidelity_validation(tmp_path):
    pytest.importorskip("neps")
    trials = pd.DataFrame(
        [
            dict(
                config_id=name,
                judge_model=name,
                agreement=agreement,
                cost_per_1k_battles=cost,
                battles_per_model=fidelity,
                status=status,
                overrides="{}",
            )
            for name, agreement, cost, fidelity, status in [
                ("cheap", 0.5, 1.0, 3, "completed"),
                ("accurate", 0.8, 2.0, 3, "completed"),
                ("dominated", 0.4, 3.0, 3, "completed"),
                ("low-fidelity", 0.9, 0.1, 1, "completed"),
                ("failed", 1.0, 0.0, 3, "failed"),
            ]
        ]
    )
    executor = SimpleNamespace(run=lambda *args, **kwargs: {"status": "completed"})
    runner._evaluate_picks(
        executor, SimpleNamespace(test_battles_per_model=None), tmp_path, trials, 3
    )
    front = pd.read_parquet(tmp_path / "pareto.parquet")
    assert set(front.config_id) == {"cheap", "accurate"}


def test_generic_route_uses_target_defaults(tmp_path):
    from judgearena.benchmarks.registry import resolve_benchmark

    tuning = _tuning_config(tmp_path).tune_judge.model_dump()
    direct = RunConfig(task=tuning["meta_eval_task"], judge={"model": "Dummy/judge"})
    outer = RunConfig(
        task="tune-judge", judge={"model": "Dummy/judge"}, tune_judge=tuning
    )
    assert outer.judge == direct.judge
    assert outer.meta_eval == direct.meta_eval
    resolved = resolve_benchmark(outer.task)
    assert resolved.adapter.name == "tune_judge" and resolved.task is None
    for target in ("unknown-task", "mt-bench"):
        with pytest.raises(ValueError):
            RunConfig(
                task="tune-judge",
                judge={"model": "Dummy/judge"},
                tune_judge={**tuning, "meta_eval_task": target},
            )
    with pytest.raises(ValueError):
        RunConfig(
            task="tune-judge",
            judge={"model": "Dummy/judge", "swap_mode": "random"},
            tune_judge=tuning,
        )
