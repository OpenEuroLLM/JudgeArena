"""Representative tuning execution, accounting, and selection contracts."""

import json
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from judgearena.config import RunConfig, dump_config, load_config
from judgearena.tuning import runner


@pytest.fixture
def config(tmp_path):
    return RunConfig(
        task="tune-judge",
        judge={"model": "Dummy/judge", "temperature": 0.0},
        run={"result_folder": str(tmp_path), "no_log_file": True},
        tune_judge={
            "meta_eval_task": "meta-eval-comparia-fr",
            "search_space": {"judge": {"temperature": [0.0, 1.0]}},
            "fidelity": {"battles_per_model": {"lower": 1, "upper": 3}},
            "neps": {"total_evaluations_to_spend": 10, "ignore_errors": True},
            "price_per_million_tokens": {"Dummy/judge": 1.0},
            "run_dir": tmp_path / "tuning",
        },
    )


def execute(path):
    cfg = load_config(path)
    assert cfg.task == "meta-eval-comparia-fr" and cfg.tune_judge is None
    assert cfg.meta_eval.split == ("test" if "test" in path.parts else "validation")
    folder = path.parent / "run"
    folder.mkdir()
    scores = dict(
        accuracy_attempted=0.6 - cfg.judge.temperature / 10,
        cohen_kappa=0.2,
        coverage=1.0,
        n_attempted=cfg.meta_eval.battles_per_model,
    )
    (folder / "results.json").write_text(
        json.dumps({"metrics": {"meta_eval_agreement": {"all": scores}}})
    )
    usage = json.dumps({"stage": "judging", "input_tokens": 2, "output_tokens": 1})
    pd.DataFrame(
        {"battle_id": ["b1", "b1"], "usage_json": [usage] * 2, "error": [None] * 2}
    ).to_parquet(folder / "annotations.parquet")


def run(config, executor=execute):
    pytest.importorskip("neps")
    return runner.run_tune_judge(config, None, execute_trial=executor)


def test_search_failure_resume_and_held_out_evaluation(config, monkeypatch):
    calls = []

    def trial(path):
        calls.append(path)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, ["judge-trial"])
        execute(path)

    results = run(config, trial)
    root = config.tune_judge.run_dir
    trials = pd.read_parquet(root / "trials.parquet")
    assert len(trials) == 10 and trials.neps_trial_id.is_unique
    assert (trials.status == "failed").sum() == 1
    final = trials[(trials.status == "completed") & (trials.battles_per_model == 3)]
    assert results.agreement.tolist() == [final.agreement.max()]
    assert set(results.status) == {"completed"}
    assert load_config(calls[-1]).meta_eval.battles_per_model == 3
    front = pd.read_parquet(root / "pareto.parquet")
    assert set(front.agreement) == {final.agreement.max()}
    monkeypatch.setattr(
        runner, "resolve_prices", lambda *a, **k: pytest.fail("repriced")
    )
    pd.testing.assert_frame_equal(results, run(config, lambda _: pytest.fail("reran")))
    config.tune_judge.search_only = True
    assert run(config, lambda _: pytest.fail("helper reran")).empty


def test_budget_end_does_not_wait_for_unstarted_batch(config, monkeypatch):
    config.tune_judge.optimizer = {
        "name": "neps_random_search",
        "ignore_fidelity": "highest_fidelity",
    }
    config.tune_judge.neps.update(total_evaluations_to_spend=2, sample_batch_size=3)
    monkeypatch.setattr(
        "judgearena.tuning.session.time.sleep", lambda _: pytest.fail("waited")
    )
    assert not run(config).empty
    assert len(pd.read_parquet(config.tune_judge.run_dir / "trials.parquet")) == 2


@pytest.mark.parametrize(
    "case,expected",
    [("cached", 0.016), ("skipped", 0.008), ("missing", None), ("agreement", None)],
)
def test_native_cost_accounting(config, tmp_path, case, expected):
    path = tmp_path / "config.yaml"
    direct = RunConfig(
        task=config.tune_judge.meta_eval_task,
        judge={"model": "Dummy/judge", "temperature": 0.0},
        meta_eval={"split": "validation"},
    )
    dump_config(direct, path)
    execute(path)
    annotations = tmp_path / "run" / "annotations.parquet"
    rows = pd.read_parquet(annotations)
    if case == "skipped":
        rows.loc[0, ["usage_json", "error"]] = [None, "context_length"]
    elif case in {"missing", "agreement"}:
        rows = rows.drop(columns=["usage_json", "error"])
    rows.to_parquet(annotations)
    if case == "missing":
        with pytest.raises(ValueError, match="fresh store_root"):
            runner._read_trial(tmp_path, runner.TokenPrice(2, 4, "test"))
    else:
        price = None if case == "agreement" else runner.TokenPrice(2, 4, "test")
        record = runner._read_trial(tmp_path, price)
        assert record["agreement"] == 0.6
        assert record["cost_per_1k_battles"] == (
            pytest.approx(expected) if expected else None
        )
        objective = runner._objective(
            {**record, "status": "completed"},
            ["agreement", "cost"] if price else ["agreement"],
        )
        assert "cost" not in objective


def test_accounting_errors_are_not_ignored(config):
    config.tune_judge.neps["total_evaluations_to_spend"] = 1

    def missing_usage(path):
        execute(path)
        pd.DataFrame({"battle_id": ["b1"]}).to_parquet(
            path.parent / "run" / "annotations.parquet"
        )

    neps = pytest.importorskip("neps")
    with pytest.raises(neps.exceptions.WorkerRaiseError) as error:
        run(config, missing_usage)
    assert isinstance(error.value.__cause__, ValueError)


def test_pareto_excludes_failed_low_fidelity_and_dominated_trials(tmp_path):
    pytest.importorskip("neps")
    trials = pd.DataFrame(
        [
            ("cheap", 0.5, 1, 3, "completed"),
            ("accurate", 0.8, 2, 3, "completed"),
            ("dominated", 0.4, 3, 3, "completed"),
            ("low", 0.9, 0.1, 1, "completed"),
            ("failed", 1, 0, 3, "failed"),
        ],
        columns="config_id agreement cost_per_1k_battles battles_per_model status".split(),
    )
    trials = trials.assign(judge_model=trials.config_id, overrides="{}")
    runner._evaluate_picks(
        SimpleNamespace(run=lambda *a, **k: {"status": "completed"}),
        SimpleNamespace(test_battles_per_model=None),
        tmp_path,
        trials,
        3,
    )
    assert set(pd.read_parquet(tmp_path / "pareto.parquet").config_id) == {
        "cheap",
        "accurate",
    }
