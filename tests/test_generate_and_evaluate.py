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
from judgearena.benchmarks.pairwise.scoring import PAIRWISE_SCORERS
from judgearena.benchmarks.registry import BenchmarkAdapter, resolve_benchmark_adapter
from judgearena.config import RunConfig
from judgearena.datasets.pairwise import PairwiseTaskData
from judgearena.models import InferenceResult
from judgearena.tasks.registry import get_packaged_task


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


def _mock_judge_response(monkeypatch, message) -> None:
    class FakeJudge:
        def batch(self, inputs, **_kwargs):
            return [message] * len(inputs)

    monkeypatch.setattr(
        benchmark_execution, "make_model", lambda **_kwargs: FakeJudge()
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
        ("mt-bench", "gpt-4"),
        ("mt-bench-official", "gpt-3.5-turbo"),
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
        "alpaca-eval",
        "arena-hard-v2.0",
        "arena-hard-v0.1",
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
            task="alpaca-eval",
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
            task="alpaca-eval",
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


def test_run_writes_roundtrippable_config(tmp_path):
    from judgearena.config import load_config

    run_pairwise(
        _cfg(
            task="alpaca-eval",
            model_A="Dummy/no answer",
            model_B="Dummy/x",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=2,
            result_folder=str(tmp_path),
        )
    )
    written = list(tmp_path.glob("*/config.yaml"))
    assert written, "config.yaml not written"
    reloaded = load_config(written[0])
    assert reloaded.task == "alpaca-eval"
    assert reloaded.model.name == "Dummy/no answer"


@pytest.mark.parametrize("swap_mode", ["fixed", "random"])
def test_arena_hard_v2_rejects_nonofficial_protocol_before_loading_data(
    monkeypatch, tmp_path, swap_mode
):
    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("task data was loaded before calibration validation")

    monkeypatch.setattr(
        generate_and_evaluate,
        "load_pairwise_task_data",
        fail_if_loaded,
    )

    with pytest.raises(ValueError, match="requires judge_swap_mode='both'"):
        run_pairwise(
            _cfg(
                task="arena-hard-v2.0-official",
                model_A="Dummy/a",
                judge_model="OpenAI/gpt-4.1",
                result_folder=str(tmp_path),
                swap_mode=swap_mode,
            )
        )


def test_arena_hard_v2_rejects_judge_input_truncation_before_loading_data(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        generate_and_evaluate,
        "load_pairwise_task_data",
        lambda *_args, **_kwargs: pytest.fail("task data loaded before validation"),
    )

    with pytest.raises(
        ValueError,
        match="requires generation.truncate_judge_input_chars=None",
    ):
        run_pairwise(
            _cfg(
                task="arena-hard-v2.0-official",
                model_A="Dummy/a",
                judge_model="OpenAI/gpt-4.1",
                result_folder=str(tmp_path),
                swap_mode="both",
                truncate_judge_input_chars=100,
            )
        )


def test_run_pairwise_judges_categories_with_their_declared_prompts(
    monkeypatch, tmp_path
):
    scored_battles = []

    def score_and_capture(battles):
        scored_battles.append(battles.copy())
        raw = PAIRWISE_SCORERS["pairwise_win_rate"].score(battles)
        return SimpleNamespace(
            summary=raw.summary,
            metrics={},
            scoring_details={},
            grouped_results={
                "category": {
                    category: PAIRWISE_SCORERS["pairwise_win_rate"]
                    .score(category_battles)
                    .summary.to_dict()
                    for category, category_battles in battles.groupby("category")
                }
            },
        )

    monkeypatch.setattr(
        generate_and_evaluate,
        "resolve_pairwise_scorer",
        lambda _name: SimpleNamespace(
            score=score_and_capture,
            check_requirements=None,
            check_runtime=None,
        ),
    )
    instructions = pd.DataFrame(
        {
            "instruction": ["q0", "q1", "q2"],
            "category": ["hard_prompt", "creative_writing", "hard_prompt"],
        },
        index=pd.Index([0, 1, 2], name="instruction_index"),
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "load_pairwise_task_data",
        lambda task, n_instructions=None: PairwiseTaskData(instructions=instructions),
    )

    prefs = run_pairwise(
        _cfg(
            task="arena-hard-v2.0-official",
            model_A="Dummy/a",
            model_B="Dummy/b",
            judge_model="Dummy/My final verdict is tie: [[A=B]]",
            n_instructions=3,
            result_folder=str(tmp_path),
            swap_mode="both",
        )
    )

    assert prefs.tolist() == [0.5] * 6
    scored = scored_battles[0]
    assert set(
        scored.loc[scored["category"] == "hard_prompt", "judge_prompt_preset"]
    ) == {"arena-hard"}
    assert set(
        scored.loc[scored["category"] == "creative_writing", "judge_prompt_preset"]
    ) == {"arena-hard-creative"}
    assert set(scored["judge_temperature"]) == {0.0}
    assert set(scored["judge_max_out_tokens"]) == {16000}
    annotations = pd.read_csv(next(tmp_path.glob("*/*annotations*.csv")))
    by_preset = annotations.groupby("prompt_preset")["instruction_index"].apply(set)
    assert by_preset.to_dict() == {
        "arena-hard": {0, 2},
        "arena-hard-creative": {1},
    }

    import json

    results = json.loads(next(tmp_path.glob("*/results-*.json")).read_text())
    assert set(results["per_category"]) == {"hard_prompt", "creative_writing"}
    run_metadata = json.loads(next(tmp_path.glob("*/run-metadata.v1.json")).read_text())
    assert {
        prompt["judge_prompt_preset"] for prompt in run_metadata["judge_prompts"]
    } == {"arena-hard", "arena-hard-creative"}


def test_run_pairwise_only_loads_category_baseline_for_assigned_rows(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        generate_and_evaluate,
        "resolve_pairwise_scorer",
        lambda _name: PAIRWISE_SCORERS["pairwise_win_rate"],
    )
    instructions = pd.DataFrame(
        {
            "instruction": ["hard", "creative"],
            "category": ["hard_prompt", "creative_writing"],
        },
        index=pd.Index(["qh", "qc"], name="instruction_index"),
    )
    model_outputs = pd.DataFrame(
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
        lambda task, n_instructions=None: PairwiseTaskData(
            instructions=instructions,
            model_outputs=model_outputs,
        ),
    )

    prefs = run_pairwise(
        _cfg(
            task="arena-hard-v2.0-official",
            model_A="candidate",
            judge_model="Dummy/My final verdict is tie: [[A=B]]",
            n_instructions=2,
            result_folder=str(tmp_path),
            swap_mode="both",
        )
    )

    assert prefs.tolist() == [0.5] * 4


def test_run_pairwise_weighted_preferences_from_judge_logprobs(monkeypatch, tmp_path):
    """The alpaca-eval preset weights verdicts by the judge's top logprobs."""
    import math

    from langchain_core.messages import AIMessage

    message = AIMessage(
        content="M",
        response_metadata={
            "logprobs": {
                "content": [
                    {
                        "token": "M",
                        "logprob": math.log(0.75),
                        "top_logprobs": [
                            {"token": "M", "logprob": math.log(0.75)},
                            {"token": "m", "logprob": math.log(0.25)},
                        ],
                    }
                ]
            }
        },
    )

    _mock_judge_response(monkeypatch, message)

    prefs = run_pairwise(
        _cfg(
            task="alpaca-eval-2.0-official",
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


@pytest.mark.parametrize("top_logprobs", [None, {}])
def test_run_pairwise_rejects_missing_required_judge_logprobs(
    monkeypatch, tmp_path, top_logprobs
):
    _mock_judge_response(
        monkeypatch,
        InferenceResult(text="M", first_token_top_logprobs=top_logprobs),
    )

    with pytest.raises(ValueError, match="required by parser 'alpaca-eval-token'"):
        run_pairwise(
            _cfg(
                task="alpaca-eval-2.0-official",
                model_A="Dummy/a",
                model_B="Dummy/b",
                judge_model="OpenRouter/fake-judge",
                n_instructions=4,
                swap_mode="random",
                result_folder=str(tmp_path),
            )
        )


def test_run_pairwise_random_swap_reorients_prefs(monkeypatch, tmp_path):
    """swap_mode='random' flips judged positions per instruction and re-orients."""
    message = InferenceResult(text="m", first_token_top_logprobs={"m": 0.0})
    _mock_judge_response(monkeypatch, message)

    prefs = run_pairwise(
        _cfg(
            task="alpaca-eval-2.0-official",
            model_A="Dummy/a",
            model_B="Dummy/b",
            judge_model="Dummy/m",
            n_instructions=4,
            swap_mode="random",
            result_folder=str(tmp_path),
        )
    )

    assert prefs.tolist() == [0.0, 0.0, 1.0, 1.0]

    annotations = pd.read_csv(next(tmp_path.glob("*/*annotations*.csv")))
    assert annotations["model_A"].tolist() == [
        "Dummy/a",
        "Dummy/a",
        "Dummy/b",
        "Dummy/b",
    ]
    assert annotations["model_B"].tolist() == [
        "Dummy/b",
        "Dummy/b",
        "Dummy/a",
        "Dummy/a",
    ]
