"""Native NePS execution and persisted trial artifacts."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from judgearena.config import RunConfig, load_config
from judgearena.tasks.registry import get_packaged_task
from judgearena.tuning import runner


def _tuning_config(tmp_path, algorithm="neps_priorband", **settings):
    multi = algorithm in {"mo_hyperband", "primo"}
    optimizer = {"name": algorithm, "eta": 3}
    if algorithm == "primo":
        optimizer.update(
            initial_design_size=2,
            prior_confidences={
                f"objective_{i}": {"judge.temperature": 0.25, "judge.model": 0.25}
                for i in range(2)
            },
            prior_centers={
                f"objective_{i}": {"judge.temperature": v, "judge.model": "Dummy/judge"}
                for i, v in enumerate((0.2, 0.8))
            },
        )
    return RunConfig(
        task="tune-judge",
        judge={"model": "Dummy/judge", "temperature": 0.2, "swap_mode": "both"},
        run={"result_folder": str(tmp_path), "no_log_file": True},
        tune_judge={
            "meta_eval_task": "meta-eval-comparia-fr",
            "neps": {"total_evaluations_to_spend": 10},
            "search_space": {
                "judge": {
                    "temperature": {"lower": 0.0, "upper": 1.0},
                    "model": ["Dummy/judge", "Dummy/other"],
                }
            },
            "fidelity": {"battles_per_model": {"lower": 1, "upper": 3}},
            "optimizer": optimizer,
            "objectives": ["agreement", "cost"] if multi else ["agreement"],
            "price_per_million_tokens": dict.fromkeys(
                ["Dummy/judge", "Dummy/other"], 1.0
            ),
            "run_dir": tmp_path / "tuning",
            **settings,
        },
    )


def _write_artifacts(folder, agreement, battles):
    folder.mkdir()
    scores = dict(
        accuracy_attempted=agreement,
        cohen_kappa=agreement,
        coverage=1.0,
        n_attempted=battles,
    )
    (folder / "results.json").write_text(
        json.dumps({"metrics": {"meta_eval_agreement": {"all": scores}}})
    )
    usage = json.dumps({"stage": "judging", "input_tokens": 2, "output_tokens": 1})
    pd.DataFrame(
        {"battle_id": ["b1", "b1"], "usage_json": [usage, usage], "error": [None, None]}
    ).to_parquet(folder / "annotations.parquet")


def _fake_trial(config_path):
    cfg = load_config(config_path)
    _write_artifacts(
        config_path.parent / "run",
        0.6 - 0.1 * cfg.judge.temperature,
        cfg.meta_eval.battles_per_model,
    )


def _run(cfg, execute=_fake_trial):
    return runner.run_tune_judge(
        cfg, get_packaged_task(cfg.task), execute_trial=execute
    )


@pytest.fixture
def artifacts(tmp_path):
    _write_artifacts(tmp_path / "run", 0.58, 1)
    return tmp_path


@pytest.mark.parametrize(
    "algorithm,fail_first",
    [
        ("neps_priorband", False),
        ("neps_hyperband", False),
        ("mo_hyperband", False),
        ("primo", False),
        ("neps_priorband", True),
        ("mo_hyperband", True),
    ],
)
def test_optimizer_searches_validation_then_tests_full_fidelity(
    tmp_path, algorithm, fail_first
):
    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path, algorithm)
    cfg.tune_judge.neps["ignore_errors"] = True
    calls = []

    def execute(path):
        trial = load_config(path)
        assert trial.task == "meta-eval-comparia-fr" and trial.tune_judge is None
        assert trial.meta_eval.split == (
            "test" if "test" in path.parts else "validation"
        )
        calls.append(path)
        if fail_first and len(calls) == 1:
            raise subprocess.CalledProcessError(1, ["judge-trial"])
        _fake_trial(path)

    results = _run(cfg, execute)
    folder = cfg.tune_judge.run_dir
    trials = pd.read_parquet(folder / "trials.parquet")
    assert len(trials) == 10 and trials.neps_trial_id.is_unique
    assert (trials.status == "failed").sum() == int(fail_first)
    final = trials[(trials.status == "completed") & (trials.battles_per_model == 3)]
    assert not results.empty and set(results.status) == {"completed"}
    assert (
        results.set_index("judge_model").agreement.to_dict()
        == final.groupby("judge_model").agreement.max().to_dict()
    )
    assert load_config(calls[-1]).meta_eval.battles_per_model == 3
    pareto = pd.read_parquet(folder / "pareto.parquet")
    if "cost" in cfg.tune_judge.objectives:
        assert set(pareto.agreement) == {final.agreement.max()}
        assert set(pareto.battles_per_model) == {3}
    else:
        assert pareto.empty


def test_session_resume_reuses_trials_and_prices(tmp_path, monkeypatch):
    pytest.importorskip("neps")
    cfg = _tuning_config(tmp_path)
    calls = []

    def execute(path):
        calls.append(path)
        _fake_trial(path)

    results = _run(cfg, execute)
    folder = cfg.tune_judge.run_dir
    trials = pd.read_parquet(folder / "trials.parquet")
    before = len(calls)
    (folder / "test_results.parquet").unlink()
    helper = _tuning_config(tmp_path, search_only=True)
    with monkeypatch.context() as patch:
        patch.setattr(
            runner,
            "resolve_prices",
            lambda *a, **kw: pytest.fail("Refetched session prices"),
        )
        resumed = _run(cfg, execute)
        assert _run(helper, execute).empty
    pd.testing.assert_frame_equal(results, resumed)
    assert len(calls) == before
    for change in (
        {"price_per_million_tokens": {"Dummy/judge": 2.0, "Dummy/other": 2.0}},
        {"meta_eval_task": "meta-eval-comparia-en"},
    ):
        with pytest.raises(ValueError, match="Incompatible configuration"):
            _run(_tuning_config(tmp_path, **change), execute)
    assert len(calls) == before
    fresh = _tuning_config(tmp_path, run_dir=tmp_path / "fresh")
    _run(fresh, execute)
    repeated = pd.read_parquet(fresh.tune_judge.run_dir / "trials.parquet")
    assert repeated.overrides.tolist() == trials.overrides.tolist()
    extended = _tuning_config(tmp_path)
    extended.tune_judge.neps["total_evaluations_to_spend"] = 11
    _run(extended, execute)
    assert len(pd.read_parquet(folder / "trials.parquet")) == 11
    _run(helper, execute)
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
    results = _run(cfg)
    assert not results.empty
    assert len(pd.read_parquet(cfg.tune_judge.run_dir / "trials.parquet")) == 2


def test_cost_counts_both_cached_orientations(artifacts):
    tmp_path = artifacts
    record = runner._read_trial(tmp_path, price=runner.TokenPrice(2.0, 4.0, "test"))
    assert record["tokens_per_battle"] == 6
    assert record["cost_per_1k_battles"] == pytest.approx(0.016)
    objective = runner._objective(
        {**record, "status": "completed"}, ["agreement", "cost"]
    )
    assert objective["objective_to_minimize"] == pytest.approx([0.42, 0.016])
    assert "cost" not in objective


def test_agreement_only_reads_old_cached_artifacts(artifacts):
    tmp_path = artifacts
    pd.DataFrame({"battle_id": ["b1"]}).to_parquet(
        tmp_path / "run" / "annotations.parquet"
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
        _run(cfg, execute)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "fresh store_root" in str(exc_info.value.__cause__)
    assert "agreement-only" in str(exc_info.value.__cause__)


def test_skipped_request_is_free_but_unparsed_response_keeps_cost(artifacts):
    tmp_path = artifacts
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
        _run(cfg, lambda _: pytest.fail("executed without prices"))


def test_pareto_report_uses_completed_full_fidelity_validation(tmp_path):
    pytest.importorskip("neps")
    trials = pd.DataFrame(
        [
            ("cheap", 0.5, 1.0, 3, "completed"),
            ("accurate", 0.8, 2.0, 3, "completed"),
            ("dominated", 0.4, 3.0, 3, "completed"),
            ("low-fidelity", 0.9, 0.1, 1, "completed"),
            ("failed", 1.0, 0.0, 3, "failed"),
        ],
        columns="config_id agreement cost_per_1k_battles battles_per_model status".split(),
    )
    trials = trials.assign(judge_model=trials.config_id, overrides="{}")
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
    for target, judge in (
        ("unknown-task", {"model": "Dummy/judge"}),
        ("mt-bench", {"model": "Dummy/judge"}),
        (tuning["meta_eval_task"], {"model": "Dummy/judge", "swap_mode": "random"}),
    ):
        with pytest.raises(ValueError):
            RunConfig(
                task="tune-judge",
                judge=judge,
                tune_judge={**tuning, "meta_eval_task": target},
            )
