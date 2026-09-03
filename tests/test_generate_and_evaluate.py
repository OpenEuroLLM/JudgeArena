import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

import judgearena.benchmarks.execution as benchmark_execution
import judgearena.benchmarks.pairwise.runner as generate_and_evaluate
import judgearena.benchmarks.registry as benchmark_registry
import judgearena.benchmarks.runner as benchmark_runner
from judgearena.benchmarks.pairwise.baselines import (
    BaselinePlan,
    native_pairwise_baseline,
    resolve_baseline_plan,
)
from judgearena.benchmarks.pairwise.runner import run_pairwise
from judgearena.benchmarks.registry import BenchmarkAdapter, resolve_benchmark_adapter
from judgearena.config import RunConfig
from judgearena.datasets.pairwise import PairwiseTaskData
from judgearena.models import InferenceResult, make_model
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import MetricSpec, ScoringSpec


def _cfg(
    *,
    task: str,
    model_A: str,
    model_B: str | None = None,
    judge_model: str,
    n_instructions: int | None = None,
    swap_mode: str = "fixed",
    result_folder: str = "results",
    truncate_judge_input_chars: int | None = None,
    max_judge_model_len: int | None = None,
    engine_kwargs: dict | None = None,
    judge_engine_kwargs: dict | None = None,
) -> RunConfig:
    return RunConfig(
        task=task,
        model={
            "name": model_A,
            "baseline": model_B,
            "engine_kwargs": engine_kwargs or {},
        },
        judge={
            "model": judge_model,
            "swap_mode": swap_mode,
            "max_model_len": max_judge_model_len,
            "engine_kwargs": judge_engine_kwargs or {},
        },
        generation={
            "n_instructions": n_instructions,
            "truncate_judge_input_chars": truncate_judge_input_chars,
        },
        run={"result_folder": result_folder},
    )


@pytest.fixture(autouse=True)
def mock_external_data_and_cache(monkeypatch):
    instructions = pd.DataFrame(
        {
            "instruction": [f"Synthetic instruction {i}" for i in range(20)],
        },
        index=pd.Index(range(20), name="instruction_index"),
    )

    monkeypatch.setattr(
        generate_and_evaluate,
        "load_pairwise_task_data",
        lambda task, n_instructions=None: PairwiseTaskData(
            instructions=(
                instructions.head(n_instructions)
                if n_instructions is not None
                else instructions
            )
        ),
    )

    def _run_without_cache(fun, **_kwargs):
        return fun()

    monkeypatch.setattr(
        generate_and_evaluate, "cache_function_dataframe", _run_without_cache
    )


def _mock_alpaca_judge(monkeypatch, message) -> dict[str, object]:
    from judgearena.benchmarks.pairwise.scoring import alpaca_eval

    captured: dict[str, object] = {}

    class FakeJudge:
        def batch(self, inputs, **_kwargs):
            return [message] * len(inputs)

    def make_fake_judge(**kwargs):
        captured.update(kwargs)
        return FakeJudge()

    monkeypatch.setattr(benchmark_execution, "make_model", make_fake_judge)
    monkeypatch.setattr(
        alpaca_eval,
        "_length_controlled_metrics",
        lambda *_args, **_kwargs: {
            "length_controlled_winrate": 50.0,
            "lc_standard_error": 1.0,
            "win_rate": 50.0,
        },
    )
    return captured


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
            ["qh", "qc"],
            categories=["hard_prompt", "creative_writing"],
        ),
    )
    assert not plan.is_single_model
    assert plan.baseline_by_index.loc["qh"] == "o3-mini-2025-01-31"
    assert plan.baseline_by_index.loc["qc"] == "gemini-2.0-flash-001"


def test_resolve_plan_alpaca_eval_uses_native_baseline():
    plan = resolve_baseline_plan(
        task_id="alpaca-eval",
        task=get_packaged_task("alpaca-eval"),
        runtime_baseline=None,
        instructions=_instructions(["q1", "q2"]),
    )
    assert plan.is_single_model
    assert plan.single_model == "gpt4_1106_preview"


def test_resolve_plan_explicit_model_b_overrides_native():
    plan = resolve_baseline_plan(
        task_id="arena-hard-v2.0",
        task=get_packaged_task("arena-hard-v2.0"),
        runtime_baseline="override",
        instructions=_instructions(
            ["q1", "q2"],
            categories=["hard_prompt", "creative_writing"],
        ),
    )
    assert plan.is_single_model
    assert plan.single_model == "override"


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("alpaca-eval", "gpt4_1106_preview"),
        ("mt-bench", "gpt-3.5-turbo"),
        ("m-arena-hard-v0.1-uk", "CohereLabs/aya-expanse-8b"),
        ("m-arena-hard-v2.0-EU", "google/gemini-2.5-flash"),
    ],
)
def test_native_pairwise_baseline_resolves_registered_tasks(task: str, expected: str):
    assert native_pairwise_baseline(task) == expected


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("alpaca-eval", "pairwise"),
        ("mt-bench", "mt_bench"),
        ("elo-comparia", "elo"),
    ],
)
def test_benchmark_adapter_resolution(task: str, expected: str):
    assert resolve_benchmark_adapter(task).name == expected


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


def test_resolve_plan_v20_unknown_category_raises():
    with pytest.raises(ValueError, match="brand_new"):
        resolve_baseline_plan(
            task_id="arena-hard-v2.0",
            task=get_packaged_task("arena-hard-v2.0"),
            runtime_baseline=None,
            instructions=_instructions(["q1"], categories=["brand_new"]),
        )


def test_baseline_plan_flat_repeats_model():
    plan = BaselinePlan.flat("b", index=pd.Index(["a", "b"]))
    assert plan.is_single_model
    assert plan.baseline_by_index.tolist() == ["b", "b"]


def test_baseline_plan_per_row_preserves_order():
    series = pd.Series(["m1", "m2"], index=["a", "b"], name="model_B")
    plan = BaselinePlan.per_row(series)
    assert not plan.is_single_model
    assert plan.unique_models == ["m1", "m2"]


@pytest.mark.parametrize(
    "task",
    [
        "alpaca-eval-ja",
        "arena-hard-v2.0-ja",
        "arena-hard-v0.1-ja",
        "fluency-french",
        "m-arena-hard-v0.1-EU",
        "m-arena-hard-v2.0-EU",
    ],
)
def test_generate_and_evaluate_context_completion(task: str, tmp_path):
    prefs = run_pairwise(
        _cfg(
            task=task,
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
            result_folder=str(tmp_path),
            # default for swap_mode is "fixed"
        )
    )

    avg_pref = sum(prefs) / len(prefs)
    assert avg_pref >= 0.9


def test_generate_and_evaluate_correct_order_bias(tmp_path):
    """Test the correction for model order bias.

    In this test, a judge that is totally biased towards model B should be corrected to be neutral.
    Since the judge favors model B regardless of the order and the completions, the average
    preference should be 0.5.
    """
    prefs = run_pairwise(
        _cfg(
            task="alpaca-eval-ja",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
            swap_mode="both",
            result_folder=str(tmp_path),
        )
    )

    avg_pref = sum(prefs) / len(prefs)
    assert avg_pref == 0.5


def test_generate_and_evaluate_passes_judge_side_controls(monkeypatch, tmp_path):
    captured = {}

    def fake_make_model(**kwargs):
        captured["make_model"] = kwargs

        class FakeJudge:
            def batch(self, inputs, **_kwargs):
                return ["score A: 0 score B: 10"] * len(inputs)

        return FakeJudge()

    monkeypatch.setattr(benchmark_execution, "make_model", fake_make_model)

    prefs = run_pairwise(
        _cfg(
            task="alpaca-eval-ja",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="VLLM/score A: 0 score B: 10",
            n_instructions=2,
            truncate_judge_input_chars=12,
            max_judge_model_len=65536,
            engine_kwargs={"tensor_parallel_size": 1},
            judge_engine_kwargs={"tensor_parallel_size": 4},
            result_folder=str(tmp_path),
        )
    )

    assert len(prefs) == 2
    assert captured["make_model"]["max_model_len"] == 65536
    assert captured["make_model"]["tensor_parallel_size"] == 4


def test_pairwise_grouping_accepts_all_canonical_battle_columns(tmp_path):
    task = get_packaged_task("alpaca-eval-ja")
    canonical_fields = (
        "model_a",
        "model_b",
        "completion_a",
        "completion_b",
        "evaluation_model",
        "source",
        "pref_hard",
    )
    protocol = task.spec.protocol.model_copy(
        update={
            "scoring": ScoringSpec(
                metrics=(
                    MetricSpec(
                        metric="pairwise_win_rate",
                        group_by=canonical_fields,
                    ),
                )
            )
        }
    )
    task = replace(task, spec=task.spec.model_copy(update={"protocol": protocol}))

    prefs = run_pairwise(
        _cfg(
            task="alpaca-eval-ja",
            model_A="Dummy/a",
            model_B="Dummy/b",
            judge_model="Dummy/score A: 10 score B: 0",
            n_instructions=2,
            result_folder=str(tmp_path),
        ),
        task,
    )

    assert len(prefs) == 2
    assert (prefs < 0.5).all()


def test_run_writes_roundtrippable_config(tmp_path):
    from judgearena.config import load_config

    run_pairwise(
        _cfg(
            task="alpaca-eval-ja",
            model_A="Dummy/no answer",
            model_B="Dummy/x",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=2,
            result_folder=str(tmp_path),
        )
    )
    written = list(tmp_path.glob("*/config.yaml"))
    assert written, "config.yaml not written"
    result = json.loads(next(tmp_path.glob("*/results-*.json")).read_text())
    assert result["metrics"]["pairwise_win_rate"]["num_battles"] == 2
    assert "winrate" not in result
    reloaded = load_config(written[0])
    assert reloaded.task == "alpaca-eval-ja"
    assert reloaded.model.name == "Dummy/no answer"


def test_run_pairwise_routes_arena_v2_baselines_and_prompts(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        generate_and_evaluate,
        "build_judge",
        lambda _cfg: make_model("Dummy/My final verdict is tie: [[A=B]]"),
    )

    def capture_battles(battles, *_args, **_kwargs):
        captured["battles"] = battles
        return {}

    monkeypatch.setattr(generate_and_evaluate, "calculate_metrics", capture_battles)
    instructions = pd.DataFrame(
        {
            "instruction": ["hard", "creative"],
            "category": ["hard_prompt", "creative_writing"],
        },
        index=pd.Index(["qh", "qc"], name="instruction_index"),
    )
    outputs = pd.DataFrame(
        {
            "instruction_index": ["qh", "qc", "qh", "qc"],
            "model": [
                "candidate",
                "candidate",
                "o3-mini-2025-01-31",
                "gemini-2.0-flash-001",
            ],
            "output": ["candidate hard", "candidate creative", "hard", "creative"],
        }
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "load_pairwise_task_data",
        lambda task, n_instructions=None: PairwiseTaskData(instructions, outputs),
    )

    run_pairwise(
        _cfg(
            task="arena-hard-v2.0",
            model_A="candidate",
            judge_model="OpenAI/gpt-4.1",
            n_instructions=2,
            result_folder=str(tmp_path),
            swap_mode="both",
        )
    )

    battles = captured["battles"]
    grouped = battles.groupby("category").first()
    assert grouped["baseline"].to_dict() == {
        "creative_writing": "gemini-2.0-flash-001",
        "hard_prompt": "o3-mini-2025-01-31",
    }
    assert grouped["judge_prompt_preset"].to_dict() == {
        "creative_writing": "arena-hard-creative",
        "hard_prompt": "arena-hard",
    }
    metadata = json.loads(next(tmp_path.glob("*/run-metadata.v1.json")).read_text())
    assert {item["judge_prompt_preset"] for item in metadata["judge_prompts"]} == {
        "arena-hard",
        "arena-hard-creative",
    }


def test_run_pairwise_weighted_preferences_from_judge_logprobs(monkeypatch, tmp_path):
    """The alpaca-eval preset weights verdicts by the judge's top logprobs."""
    import math

    message = InferenceResult(
        text="M",
        first_token_top_logprobs={
            "M": math.log(0.75),
            "m": math.log(0.25),
        },
    )

    judge_kwargs = _mock_alpaca_judge(monkeypatch, message)

    prefs = run_pairwise(
        _cfg(
            task="alpaca-eval",
            model_A="Dummy/a",
            model_B="Dummy/b",
            judge_model="OpenRouter/fake-judge",
            n_instructions=4,
            swap_mode="random",
            result_folder=str(tmp_path),
        )
    )

    # Judged pref is P(M)=0.75 everywhere; unswitched rows (2, 3 under the
    # golden mask) show the baseline in slot A and re-orient to 0.25.
    assert prefs.tolist() == pytest.approx([0.75, 0.75, 0.25, 0.25])
    results = json.loads(next(tmp_path.glob("*/results-*.json")).read_text())
    assert "alpaca_eval_length_controlled" in results["metrics"]
    assert judge_kwargs["top_logprobs"] == 5
