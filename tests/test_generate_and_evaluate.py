from types import SimpleNamespace

import pandas as pd
import pytest

import judgearena.benchmarks.execution as benchmark_execution
import judgearena.benchmarks.pairwise.runner as generate_and_evaluate
import judgearena.benchmarks.registry as benchmark_registry
import judgearena.benchmarks.runner as benchmark_runner
from judgearena.benchmarks.pairwise.baselines import (
    native_pairwise_baseline,
    resolve_baseline_plan,
)
from judgearena.benchmarks.pairwise.runner import run_pairwise
from judgearena.benchmarks.registry import BenchmarkAdapter, resolve_benchmark_adapter
from judgearena.config import RunConfig
from judgearena.datasets.pairwise import PairwiseTaskData
from judgearena.tasks.registry import get_packaged_task


@pytest.fixture
def cfg(tmp_path):
    return RunConfig(
        task="alpaca-eval",
        model={"name": "Dummy/no answer", "baseline": "Dummy/x"},
        judge={"model": "Dummy/score A: 0 score B: 10", "swap_mode": "fixed"},
        generation={"n_instructions": 2},
        run={"result_folder": str(tmp_path)},
    )


@pytest.fixture(autouse=True)
def mock_external_data_and_cache(monkeypatch):
    instructions = pd.DataFrame(
        {"instruction": ["Synthetic instruction 0", "Synthetic instruction 1"]},
        index=pd.Index(range(2), name="instruction_index"),
    )

    monkeypatch.setattr(
        generate_and_evaluate,
        "load_pairwise_task_data",
        lambda task, n_instructions=None: PairwiseTaskData(instructions=instructions),
    )

    def _run_without_cache(fun, **_kwargs):
        return fun()

    monkeypatch.setattr(
        generate_and_evaluate, "cache_function_dataframe", _run_without_cache
    )


def _instructions(ids: list[str], categories: list[str] | None = None) -> pd.DataFrame:
    data = {"instruction": list(ids)}
    if categories is not None:
        data["category"] = list(categories)
    return pd.DataFrame(data, index=pd.Index(ids, name="instruction_index"))


def test_resolve_plan_v01_flat_default():
    plan = resolve_baseline_plan(
        task_id="arena-hard-v0.1",
        task=get_packaged_task("arena-hard-v0.1"),
        runtime_baseline=None,
        instructions=_instructions(["q1", "q2"]),
    )
    assert plan.is_single_model
    assert plan.single_model == "gpt-4-0314"


def test_resolve_plan_v20_routes_per_category():
    plan = resolve_baseline_plan(
        task_id="arena-hard-v2.0",
        task=get_packaged_task("arena-hard-v2.0"),
        runtime_baseline=None,
        instructions=_instructions(
            ["qh", "qc"], categories=["hard_prompt", "creative_writing"]
        ),
    )
    assert not plan.is_single_model
    assert plan.baseline_by_index.loc["qh"] == "o3-mini-2025-01-31"
    assert plan.baseline_by_index.loc["qc"] == "gemini-2.0-flash-001"


def test_resolve_plan_explicit_model_b_overrides_native():
    plan = resolve_baseline_plan(
        task_id="arena-hard-v2.0",
        task=get_packaged_task("arena-hard-v2.0"),
        runtime_baseline="override",
        instructions=_instructions(
            ["q1", "q2"], categories=["hard_prompt", "creative_writing"]
        ),
    )
    assert plan.is_single_model
    assert plan.single_model == "override"


def test_native_pairwise_baseline_resolves_registered_task():
    assert (
        native_pairwise_baseline("m-arena-hard-v0.1-uk") == "CohereLabs/aya-expanse-8b"
    )


def test_benchmark_adapter_resolution():
    assert resolve_benchmark_adapter("elo-comparia").name == "elo"


def test_registered_task_runner_wins_over_legacy_fallback(monkeypatch):
    fallback = BenchmarkAdapter("fallback", None, lambda _cfg: None)
    pairwise = BenchmarkAdapter("pairwise", frozenset(), lambda _cfg: None)
    resolved = SimpleNamespace(
        spec=SimpleNamespace(protocol=SimpleNamespace(runner="pairwise"))
    )
    monkeypatch.setattr(
        benchmark_registry, "benchmark_adapters", lambda: (fallback, pairwise)
    )
    monkeypatch.setattr(benchmark_registry, "get_packaged_task", lambda _task: resolved)

    assert benchmark_registry.resolve_benchmark_adapter("yaml-task") is pairwise


def test_benchmark_dispatch_passes_the_resolved_task(monkeypatch):
    resolved = SimpleNamespace(
        spec=SimpleNamespace(protocol=SimpleNamespace(runner="pairwise"))
    )
    captured = {}
    pairwise = BenchmarkAdapter(
        "pairwise",
        frozenset(),
        lambda cfg, task: captured.update(cfg=cfg, task=task) or "result",
    )
    monkeypatch.setattr(benchmark_registry, "benchmark_adapters", lambda: (pairwise,))
    monkeypatch.setattr(benchmark_registry, "get_packaged_task", lambda _task: resolved)
    cfg = SimpleNamespace(task="yaml-task")

    result = benchmark_runner.run_benchmark(cfg)

    assert result == "result"
    assert captured == {"cfg": cfg, "task": resolved}


def test_resolve_plan_task_without_native_baseline_requires_model_b():
    with pytest.raises(ValueError, match="baseline"):
        resolve_baseline_plan(
            task_id="fluency-french",
            task=None,
            runtime_baseline=None,
            instructions=_instructions(["q1"]),
        )


def test_resolve_plan_v20_missing_category_raises():
    with pytest.raises(ValueError, match="category"):
        resolve_baseline_plan(
            task_id="arena-hard-v2.0",
            task=get_packaged_task("arena-hard-v2.0"),
            runtime_baseline=None,
            instructions=_instructions(["q1"]),
        )


def test_generate_and_evaluate_context_completion(cfg):
    prefs = run_pairwise(cfg)
    assert sum(prefs) / len(prefs) >= 0.9


def test_generate_and_evaluate_correct_order_bias(cfg):
    """Swapping neutralizes a judge that always favors model B."""
    cfg.judge.swap_mode = "both"
    prefs = run_pairwise(cfg)
    assert sum(prefs) / len(prefs) == 0.5


def test_generate_and_evaluate_passes_judge_side_controls(monkeypatch, cfg):
    captured = {}

    def fake_make_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            batch=lambda inputs, **_: ["score A: 0 score B: 10"] * len(inputs)
        )

    monkeypatch.setattr(benchmark_execution, "make_model", fake_make_model)
    cfg.judge.model = "VLLM/judge"
    cfg.judge.max_model_len = 65536
    cfg.model.engine_kwargs = {"tensor_parallel_size": 1}
    cfg.judge.engine_kwargs = {"tensor_parallel_size": 4}
    prefs = run_pairwise(cfg)

    assert len(prefs) == 2
    assert captured["max_model_len"] == 65536
    assert captured["tensor_parallel_size"] == 4


def test_run_writes_roundtrippable_config(cfg, tmp_path):
    from judgearena.config import load_config

    run_pairwise(cfg)
    written = list(tmp_path.glob("*/config.yaml"))
    assert written, "config.yaml not written"
    reloaded = load_config(written[0])
    assert reloaded.task == "alpaca-eval"
    assert reloaded.model.name == "Dummy/no answer"
