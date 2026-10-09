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


def _tuning_config(tmp_path, algorithm="neps_priorband", **settings):
    optimizer = {"name": algorithm, "eta": 3}
    objectives = ["agreement"]
    if algorithm in {"mo_hyperband", "primo"}:
        objectives.append("cost")
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
        "judge": {
            "temperature": {"lower": 0.0, "upper": 1.0},
            "model": {
                "choices": ["Dummy/judge", "Dummy/other"],
                "prior": "Dummy/other",
            },
        }
    }
    return RunConfig(
        task="tune-judge",
        judge={"model": "Dummy/judge", "temperature": 0.2, "swap_mode": "both"},
        run={"result_folder": str(tmp_path), "no_log_file": True},
        tune_judge={
            "meta_eval_task": "meta-eval-comparia-fr",
            "neps": {
                "total_evaluations_to_spend": 10,
            },
            "search_space": specs,
            "fidelity": {"battles_per_model": {"lower": 1, "upper": 3}},
            "optimizer": optimizer,
            "objectives": objectives,
            "price_per_million_tokens": {"Dummy/judge": 1.0, "Dummy/other": 1.0},
            "run_dir": tmp_path / "tuning",
            **settings,
        },
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
            "usage_json": [
                json.dumps({"stage": "judging", "input_tokens": 2, "output_tokens": 1}),
                json.dumps({"stage": "judging", "input_tokens": 2, "output_tokens": 1}),
            ],
            "error": [None, None],
            "cache_hit": [True, True],
        }
    ).to_parquet(folder / "annotations.parquet")


@pytest.mark.parametrize(
    "algorithm", ["neps_priorband", "neps_hyperband", "mo_hyperband", "primo"]
)
def test_native_optimizer_smoke(tmp_path, algorithm, monkeypatch):
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
    helper = _tuning_config(tmp_path, search_only=True)
    with monkeypatch.context() as patch:
        patch.setattr(
            runner,
            "resolve_prices",
            lambda *a, **kw: pytest.fail("Refetched session prices"),
        )
        resumed = runner.run_tune_judge(cfg, task, execute_trial=execute)
        assert runner.run_tune_judge(helper, task, execute_trial=execute).empty
    pd.testing.assert_frame_equal(results, resumed)
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


def test_budget_end_does_not_wait_for_unstarted_batch(tmp_path, monkeypatch):
    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path)
    cfg.tune_judge.optimizer = {
        "name": "neps_random_search",
        "ignore_fidelity": "highest_fidelity",
    }
    cfg.tune_judge.neps.update(
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


def test_cost_counts_both_cached_orientations(tmp_path):
    path = tmp_path / "config.yaml"
    dump_config(_tuning_config(tmp_path), path)
    _fake_trial(path)
    record = runner._read_trial(tmp_path, price=runner.TokenPrice(2.0, 4.0, "test"))
    assert record["tokens_per_battle"] == 6
    assert record["cost_per_1k_battles"] == pytest.approx(0.016)
    objective = runner._objective(
        {**record, "status": "completed"}, ["agreement", "cost"]
    )
    assert objective["objective_to_minimize"] == pytest.approx([0.42, 0.016])
    assert "cost" not in objective


def test_agreement_only_reads_old_cached_artifacts(tmp_path):
    path = tmp_path / "config.yaml"
    dump_config(_tuning_config(tmp_path), path)
    _fake_trial(path)
    annotations_path = tmp_path / "run" / "annotations.parquet"
    pd.read_parquet(annotations_path).drop(columns=["usage_json", "error"]).to_parquet(
        annotations_path, index=False
    )

    result = runner._read_trial(tmp_path, price=None)
    assert result["agreement"] == pytest.approx(0.58)
    assert result["tokens_per_battle"] is None
    assert result["cost_per_1k_battles"] is None


def test_missing_native_usage_aborts_even_when_failed_trials_are_ignored(tmp_path):
    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path, "mo_hyperband")
    cfg.tune_judge.neps.update(ignore_errors=True, total_evaluations_to_spend=1)

    def execute(path):
        _fake_trial(path)
        annotations_path = path.parent / "run" / "annotations.parquet"
        annotations = pd.read_parquet(annotations_path)
        annotations["usage_json"] = None
        annotations.to_parquet(annotations_path, index=False)

    from neps.exceptions import WorkerRaiseError

    with pytest.raises(WorkerRaiseError) as exc_info:
        runner.run_tune_judge(cfg, get_packaged_task(cfg.task), execute_trial=execute)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "fresh store_root" in str(exc_info.value.__cause__)
    assert "agreement-only" in str(exc_info.value.__cause__)


def test_skipped_request_is_free_but_unparsed_response_keeps_cost(tmp_path):
    path = tmp_path / "config.yaml"
    dump_config(_tuning_config(tmp_path), path)
    _fake_trial(path)
    annotations_path = tmp_path / "run" / "annotations.parquet"
    annotations = pd.read_parquet(annotations_path)
    annotations.loc[0, ["usage_json", "error"]] = [None, "context_length"]
    annotations.loc[1, "judge_completion"] = "unparseable"
    annotations.to_parquet(annotations_path, index=False)
    record = runner._read_trial(tmp_path, runner.TokenPrice(2, 4, "test"))
    assert record["tokens_per_battle"] == 3
    assert record["cost_per_1k_battles"] == pytest.approx(0.008)


def test_prices_required_before_execution(tmp_path, monkeypatch):
    from judgearena import pricing

    monkeypatch.setattr(pricing, "_fetch_openrouter", lambda _: [])

    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path, "mo_hyperband", price_per_million_tokens={})
    with pytest.raises(ValueError, match="No token price"):
        runner.run_tune_judge(
            cfg,
            get_packaged_task(cfg.task),
            execute_trial=lambda _: pytest.fail("executed without prices"),
        )


@pytest.mark.parametrize("algorithm", ["neps_priorband", "mo_hyperband"])
def test_failed_trial_does_not_stop_search(tmp_path, algorithm):
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
    if algorithm == "neps_priorband":
        assert pareto.empty
        assert "cost" not in cfg.tune_judge.objectives
        return
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
    assert resolved.adapter.name == "tune_judge" and resolved.task.spec.dataset is None
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
