import pandas as pd
import pytest

from judgearena.generate_and_evaluate import (
    BaselinePlan,
    CliArgs,
    _resolve_baseline_plan,
)


def _make_args(dataset, model_b=None):
    return CliArgs(
        dataset=dataset,
        model_A="A",
        model_B=model_b,
        judge_model="J",
    )


def _instructions(ids, categories=None):
    data = {"instruction": list(ids)}
    if categories is not None:
        data["category"] = list(categories)
    return pd.DataFrame(data, index=pd.Index(ids, name="instruction_index"))


def test_resolve_plan_v01_flat_default():
    plan = _resolve_baseline_plan(
        args=_make_args("arena-hard-v0.1"),
        instructions_df=_instructions(["q1", "q2"]),
    )
    assert plan.is_flat
    assert plan.single_model == "gpt-4-0314"


def test_resolve_plan_v20_routes_per_category():
    plan = _resolve_baseline_plan(
        args=_make_args("arena-hard-v2.0"),
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
        args=_make_args("arena-hard-v2.0", model_b="override"),
        instructions_df=_instructions(
            ["q1", "q2"], categories=["hard_prompt", "creative_writing"]
        ),
    )
    assert plan.is_flat
    assert plan.single_model == "override"


def test_resolve_plan_non_arena_hard_requires_model_b():
    with pytest.raises(ValueError, match="model_B"):
        _resolve_baseline_plan(
            args=_make_args("alpaca-eval"),
            instructions_df=_instructions(["q1"]),
        )


def test_resolve_plan_v20_missing_category_raises():
    with pytest.raises(ValueError, match="category"):
        _resolve_baseline_plan(
            args=_make_args("arena-hard-v2.0"),
            instructions_df=_instructions(["q1"]),
        )


def test_resolve_plan_v20_unknown_category_raises():
    with pytest.raises(ValueError, match="brand_new"):
        _resolve_baseline_plan(
            args=_make_args("arena-hard-v2.0"),
            instructions_df=_instructions(["q1"], categories=["brand_new"]),
        )


def test_baseline_plan_flat_repeats_model():
    plan = BaselinePlan.flat("b", index=pd.Index(["a", "b"]))
    assert plan.is_flat
    assert plan.baseline_by_index.tolist() == ["b", "b"]


def test_baseline_plan_per_row_preserves_order():
    series = pd.Series(["m1", "m2"], index=["a", "b"], name="model_B")
    plan = BaselinePlan.per_row(series)
    assert not plan.is_flat
    assert plan.unique_models == ["m1", "m2"]
