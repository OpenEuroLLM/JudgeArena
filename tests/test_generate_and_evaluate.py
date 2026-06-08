import pandas as pd
import pytest

import judgearena.generate_and_evaluate as generate_and_evaluate
from judgearena.config import RunConfig
from judgearena.generate_and_evaluate import (
    BaselinePlan,
    _resolve_baseline_plan,
    native_pairwise_baseline,
)
from judgearena.generate_and_evaluate import (
    main as main_generate_and_eval,
)


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
        model={"path": model_A, "path_b": model_B, "engine_kwargs": engine_kwargs or {}},
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
        "load_instructions",
        lambda dataset, n_instructions=None: (
            instructions.head(n_instructions)
            if n_instructions is not None
            else instructions
        ),
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "load_contexts",
        lambda dataset: instructions.loc[:, "instruction"],
    )

    monkeypatch.setattr(
        generate_and_evaluate,
        "try_load_dataset_completions",
        lambda dataset, model, n_instructions: None,
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
    plan = _resolve_baseline_plan(
        task="arena-hard-v0.1",
        model_b=None,
        instructions_df=_instructions(["q1", "q2"]),
    )
    assert plan.is_flat
    assert plan.single_model == "gpt-4-0314"


def test_resolve_plan_v20_routes_per_category():
    plan = _resolve_baseline_plan(
        task="arena-hard-v2.0",
        model_b=None,
        instructions_df=_instructions(
            ["qh", "qc"],
            categories=["hard_prompt", "creative_writing"],
        ),
    )
    assert not plan.is_flat
    assert plan.baseline_by_index.loc["qh"] == "o3-mini-2025-01-31"
    assert plan.baseline_by_index.loc["qc"] == "gemini-2.0-flash-001"


def test_resolve_plan_explicit_model_b_overrides_native():
    plan = _resolve_baseline_plan(
        task="arena-hard-v2.0",
        model_b="override",
        instructions_df=_instructions(
            ["q1", "q2"],
            categories=["hard_prompt", "creative_writing"],
        ),
    )
    assert plan.is_flat
    assert plan.single_model == "override"


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("alpaca-eval", "gpt4_1106_preview"),
        ("mt-bench", "gpt-4"),
        ("m-arena-hard-v0.1-uk", "CohereLabs/aya-expanse-8b"),
        ("m-arena-hard-v2.0-EU", "google/gemini-2.5-flash"),
    ],
)
def test_native_pairwise_baseline_resolves_registered_tasks(task: str, expected: str):
    assert native_pairwise_baseline(task) == expected


def test_resolve_plan_task_without_native_baseline_requires_model_b():
    with pytest.raises(ValueError, match="model_B"):
        _resolve_baseline_plan(
            task="fluency-french",
            model_b=None,
            instructions_df=_instructions(["q1"]),
        )


def test_resolve_plan_v20_missing_category_raises():
    with pytest.raises(ValueError, match="category"):
        _resolve_baseline_plan(
            task="arena-hard-v2.0",
            model_b=None,
            instructions_df=_instructions(["q1"]),
        )


def test_baseline_plan_per_row_preserves_order():
    series = pd.Series(["m1", "m2"], index=["a", "b"], name="model_B")
    plan = BaselinePlan.per_row(series)
    assert not plan.is_flat
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
    prefs = main_generate_and_eval(
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
    prefs = main_generate_and_eval(
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

    monkeypatch.setattr(generate_and_evaluate, "make_model", fake_make_model)

    prefs = main_generate_and_eval(
        _cfg(
            task="alpaca-eval",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
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

    main_generate_and_eval(
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
    assert reloaded.model.path == "Dummy/no answer"
