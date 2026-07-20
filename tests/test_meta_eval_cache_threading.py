"""Unified inference cache integration tests for meta-evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import judgearena.evaluate as evaluate_module
import judgearena.meta_eval.annotate as meta_annotate
import judgearena.meta_eval.runner as meta_eval_runner
import judgearena.models as models_module
from judgearena.config import (
    CacheArgs,
    RunConfig,
    inference_cache_task,
    meta_eval_cache_task,
)
from judgearena.inference_cache import InferenceCache
from judgearena.meta_eval.cli_args import CliMetaEvalArgs, meta_eval_args_from_config
from judgearena.meta_eval.prompts import PromptModeSpec
from judgearena.meta_eval.runner import main as meta_eval_main
from judgearena.models import make_model
from judgearena.store_sqlite import SQLiteInferenceStore, descriptor_hash, store_folder


def _meta_args_with_cache(tmp_path: Path, **overrides) -> CliMetaEvalArgs:
    values = {
        "reference_arena": "LMArena-140k",
        "prompt_mode": "standard",
        "top_models": 3,
        "battles_per_model": 1,
        "batch_size": 8,
        "languages": ["en"],
        "judge_model": "Dummy/score_A: 9\nscore_B: 1",
        "result_folder": str(tmp_path / "results"),
        "no_log_file": True,
        "cache": CacheArgs(store_root=str(tmp_path / "cache")),
    }
    values.update(overrides)
    return CliMetaEvalArgs(**values)


def _prompt_spec() -> PromptModeSpec:
    return PromptModeSpec(
        name="standard",
        system_prompt="system",
        user_prompt_template=(
            "Question: {user_prompt}\nA: {completion_A}\nB: {completion_B}"
        ),
    )


def _single_battle_frame() -> pd.DataFrame:
    conv_a = [
        {"role": "user", "content": "Question 0"},
        {"role": "assistant", "content": "Answer A 0"},
    ]
    conv_b = [
        {"role": "user", "content": "Question 0"},
        {"role": "assistant", "content": "Answer B 0"},
    ]
    return pd.DataFrame(
        [
            {
                "question_id": "q-0",
                "model_a": "model-0",
                "model_b": "model-1",
                "winner": "model_a",
                "lang": "en",
                "benchmark": "LMArena-140k",
                "conversation_a": conv_a,
                "conversation_b": conv_b,
            }
        ]
    )


def _base_meta_eval_payload(tmp_path: Path, **overrides) -> dict:
    payload = {
        "task": "meta-eval",
        "judge": {"model": "Dummy/j"},
        "run": {"result_folder": str(tmp_path / "results"), "no_log_file": True},
    }
    payload.update(overrides)
    return payload


def test_meta_eval_args_from_config_carries_cache(tmp_path):
    cfg = RunConfig(
        **_base_meta_eval_payload(
            tmp_path,
            cache={"store_root": str(tmp_path / "cache"), "cache_mode": "refresh"},
        )
    )
    args = meta_eval_args_from_config(cfg)
    assert args.cache.store_root == str(tmp_path / "cache")
    assert args.cache.cache_mode == "refresh"


def test_meta_eval_args_from_config_carries_strip_thinking(tmp_path):
    cfg = RunConfig(
        **_base_meta_eval_payload(
            tmp_path,
            judge={
                "model": "Dummy/j",
                "strip_thinking_before_judging": True,
            },
        )
    )

    args = meta_eval_args_from_config(cfg)

    assert args.strip_thinking_before_judging is True


def test_inference_cache_task_includes_reference_arena(tmp_path):
    cfg = RunConfig(**_base_meta_eval_payload(tmp_path))
    assert inference_cache_task(cfg) == "meta-eval-LMArena-140k"
    assert meta_eval_cache_task("LMArena-140k") == "meta-eval-LMArena-140k"


def test_meta_eval_forwards_strip_thinking_to_annotation(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def spy_annotate_battles(*, instructions, **kwargs):
        captured.update(kwargs)
        return [
            evaluate_module.JudgeAnnotation(
                judge_completion="score A: 9 score B: 1",
                instruction=instruction,
                completion_A="A",
                completion_B="B",
                judge_input="rendered",
            )
            for instruction in instructions
        ]

    monkeypatch.setattr(meta_annotate, "annotate_battles", spy_annotate_battles)
    args = _meta_args_with_cache(tmp_path, strip_thinking_before_judging=True)

    meta_annotate.annotate_sample(
        _single_battle_frame(),
        args,
        judge_chat_model=object(),
        prompt_spec=_prompt_spec(),
    )

    assert captured["strip_thinking_before_judging"] is True


def test_meta_eval_second_run_reuses_cached_rows(monkeypatch, tmp_path):
    uncached_calls = {"count": 0}
    real_uncached = models_module._do_inference_uncached

    def counting_uncached(*args, **kwargs):
        uncached_calls["count"] += 1
        return real_uncached(*args, **kwargs)

    monkeypatch.setattr(models_module, "_do_inference_uncached", counting_uncached)

    args = _meta_args_with_cache(tmp_path, swap_mode="fixed")
    sample = _single_battle_frame()
    prompt_spec = _prompt_spec()
    judge = make_model(args.judge_model, max_tokens=32)

    with InferenceCache(
        args.cache.store_root,
        meta_eval_cache_task(args.reference_arena),
        mode="use",
    ) as cache:
        meta_annotate.annotate_sample(
            sample,
            args,
            judge_chat_model=judge,
            prompt_spec=prompt_spec,
            cache=cache,
        )
    assert uncached_calls["count"] > 0

    uncached_calls["count"] = 0
    with InferenceCache(
        args.cache.store_root,
        meta_eval_cache_task(args.reference_arena),
        mode="use",
    ) as cache:
        meta_annotate.annotate_sample(
            sample,
            args,
            judge_chat_model=judge,
            prompt_spec=prompt_spec,
            cache=cache,
        )
    assert uncached_calls["count"] == 0


def test_meta_eval_swapped_orientations_store_distinct_associations(
    monkeypatch, tmp_path
):
    captured_metadata: list[dict] = []
    real_do_inference = evaluate_module.do_inference

    def spy_do_inference(*args, **kwargs):
        cache_meta = kwargs.get("cache_meta")
        if cache_meta is not None:
            captured_metadata.extend(cache_meta.get("metadata", []))
        return real_do_inference(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "do_inference", spy_do_inference)

    args = _meta_args_with_cache(tmp_path, swap_mode="both")
    prompt_spec = _prompt_spec()
    judge = make_model(args.judge_model, max_tokens=32)

    with InferenceCache(
        args.cache.store_root,
        meta_eval_cache_task(args.reference_arena),
        mode="use",
    ) as cache:
        meta_annotate.annotate_sample(
            _single_battle_frame(),
            args,
            judge_chat_model=judge,
            prompt_spec=prompt_spec,
            cache=cache,
        )

    orientations = {row["orientation"] for row in captured_metadata}
    assert orientations == {"forward", "swapped"}
    assert all(row["question_id"] == "q-0" for row in captured_metadata)
    assert all(row["reference_arena"] == "LMArena-140k" for row in captured_metadata)

    model = make_model(args.judge_model, max_tokens=32)
    descriptor = model.cache_descriptor()
    folder = store_folder(
        tmp_path / "cache",
        meta_eval_cache_task(args.reference_arena),
        model.model_spec,
        descriptor_hash(descriptor),
    )
    with SQLiteInferenceStore(folder / "inference.db") as store:
        metadata_rows = store.query_metadata()
    stored_orientations = {
        json.loads(row["metadata_json"])["orientation"]
        for _, row in metadata_rows.iterrows()
    }
    assert stored_orientations == {"forward", "swapped"}


def test_meta_eval_changed_rendered_input_invalidates_only_that_row(
    monkeypatch, tmp_path
):
    uncached_calls = {"count": 0}
    real_uncached = models_module._do_inference_uncached

    def counting_uncached(*args, **kwargs):
        uncached_calls["count"] += 1
        return real_uncached(*args, **kwargs)

    monkeypatch.setattr(models_module, "_do_inference_uncached", counting_uncached)

    args = _meta_args_with_cache(tmp_path)
    prompt_spec = _prompt_spec()
    judge = make_model(args.judge_model, max_tokens=32)
    original = _single_battle_frame()
    changed = original.copy()
    changed.iloc[0, changed.columns.get_loc("conversation_a")] = [
        {"role": "user", "content": "Changed question"},
        {"role": "assistant", "content": "Answer A 0"},
    ]

    with InferenceCache(
        args.cache.store_root,
        meta_eval_cache_task(args.reference_arena),
        mode="use",
    ) as cache:
        meta_annotate.annotate_sample(
            original,
            args,
            judge_chat_model=judge,
            prompt_spec=prompt_spec,
            cache=cache,
        )
    first_count = uncached_calls["count"]
    assert first_count > 0

    uncached_calls["count"] = 0
    with InferenceCache(
        args.cache.store_root,
        meta_eval_cache_task(args.reference_arena),
        mode="use",
    ) as cache:
        meta_annotate.annotate_sample(
            changed,
            args,
            judge_chat_model=judge,
            prompt_spec=prompt_spec,
            cache=cache,
        )
    assert uncached_calls["count"] == first_count


def test_meta_eval_parsing_and_costs_recompute_from_cached_output(
    monkeypatch, tmp_path
):
    args = _meta_args_with_cache(tmp_path)
    prompt_spec = PromptModeSpec(
        name="standard",
        system_prompt="system",
        user_prompt_template="user",
    )
    judge = make_model("Dummy/score_A: 1\nscore_B: 9", max_tokens=32)
    sample = _single_battle_frame()

    with InferenceCache(
        args.cache.store_root,
        meta_eval_cache_task(args.reference_arena),
        mode="use",
    ) as cache:
        first = meta_annotate.annotate_sample(
            sample,
            args,
            judge_chat_model=judge,
            prompt_spec=prompt_spec,
            cache=cache,
        )

    cost_calls = {"count": 0}
    original_cost = meta_annotate.estimate_annotation_cost_usd

    def spy_cost(*, judge_input, judge_completion, judge_model):
        cost_calls["count"] += 1
        return original_cost(
            judge_input=judge_input,
            judge_completion=judge_completion,
            judge_model=judge_model,
        )

    monkeypatch.setattr(meta_annotate, "estimate_annotation_cost_usd", spy_cost)

    with InferenceCache(
        args.cache.store_root,
        meta_eval_cache_task(args.reference_arena),
        mode="use",
    ) as cache:
        second = meta_annotate.annotate_sample(
            sample,
            args,
            judge_chat_model=judge,
            prompt_spec=prompt_spec,
            cache=cache,
        )

    assert first.iloc[0]["winner_llm"] == "model_b"
    assert second.iloc[0]["winner_llm"] == "model_b"
    assert cost_calls["count"] > 0


def test_meta_eval_runner_uses_one_shared_cache_handle(
    monkeypatch, tmp_path, synthetic_arena_df
):
    captured: list[object] = []

    def spy_annotate_sample(df_sample, args, *, cache=None, **kwargs):
        captured.append(cache)
        return pd.DataFrame(
            {
                "question_id": df_sample["question_id"],
                "model_a": df_sample["model_a"],
                "model_b": df_sample["model_b"],
                "winner": df_sample["winner"],
                "lang": df_sample["lang"],
                "benchmark": df_sample["benchmark"],
                "orientation": "forward",
                "instruction": "instr",
                "completion_a": "A",
                "completion_b": "B",
                "judge_input": "prompt",
                "judge_completion": "score_A: 9\nscore_B: 1",
                "estimated_input_tokens": 2,
                "estimated_output_tokens": 5,
                "cost_usd": 0.001,
                "cost_source": "estimated",
                "winner_llm": df_sample["winner"],
                "pref_llm": 0.0,
            }
        )

    monkeypatch.setattr(
        meta_eval_runner,
        "load_reference_arena_battles",
        lambda reference_arena, languages=None: synthetic_arena_df,
    )
    monkeypatch.setattr(
        meta_eval_runner,
        "select_top_models",
        lambda df, top_models: (["model-0", "model-1"], df),
    )
    monkeypatch.setattr(
        meta_eval_runner,
        "sample_battles_per_model",
        lambda df_top, models, battles_per_model, seed: df_top.head(1),
    )
    monkeypatch.setattr(meta_eval_runner, "make_model", lambda **_kwargs: object())
    monkeypatch.setattr(meta_eval_runner, "annotate_sample", spy_annotate_sample)

    args = _meta_args_with_cache(tmp_path, top_models=2)
    meta_eval_main(args)

    assert len(captured) == 1
    assert captured[0] is not None


def test_meta_eval_args_json_includes_cache_config(
    tmp_path, monkeypatch, synthetic_arena_df
):
    monkeypatch.setattr(
        meta_eval_runner,
        "load_reference_arena_battles",
        lambda reference_arena, languages=None: synthetic_arena_df,
    )
    monkeypatch.setattr(
        meta_eval_runner,
        "select_top_models",
        lambda df, top_models: (["model-0", "model-1"], df),
    )
    monkeypatch.setattr(
        meta_eval_runner,
        "sample_battles_per_model",
        lambda df_top, models, battles_per_model, seed: df_top.head(1),
    )
    monkeypatch.setattr(meta_eval_runner, "make_model", lambda **_kwargs: object())
    monkeypatch.setattr(
        meta_eval_runner,
        "annotate_sample",
        lambda df_sample, args, **kwargs: pd.DataFrame(
            {
                "question_id": df_sample["question_id"],
                "model_a": df_sample["model_a"],
                "model_b": df_sample["model_b"],
                "winner": df_sample["winner"],
                "lang": df_sample["lang"],
                "benchmark": df_sample["benchmark"],
                "orientation": "forward",
                "instruction": "instr",
                "completion_a": "A",
                "completion_b": "B",
                "judge_input": "prompt",
                "judge_completion": "score_A: 9\nscore_B: 1",
                "estimated_input_tokens": 2,
                "estimated_output_tokens": 5,
                "cost_usd": 0.001,
                "cost_source": "estimated",
                "winner_llm": df_sample["winner"],
                "pref_llm": 0.0,
            }
        ),
    )

    args = _meta_args_with_cache(tmp_path, top_models=2)
    meta_eval_main(args)
    output_dir = next(Path(args.result_folder).glob("meta-eval-*"))
    args_payload = json.loads((output_dir / "args.json").read_text(encoding="utf-8"))
    assert args_payload["cache"]["store_root"] == str(tmp_path / "cache")
    assert "ignore_cache" not in args_payload


def test_meta_eval_args_serialization_redacts_engine_secrets(tmp_path):
    args = _meta_args_with_cache(tmp_path)
    args.engine_kwargs = {
        "temperature": 0.2,
        "api_key": "must-not-leak",
        "default_headers": {"Authorization": "secret"},
    }

    payload = args.to_jsonable()

    assert payload["engine_kwargs"] == {"temperature": 0.2}
    assert "must-not-leak" not in json.dumps(payload)
    assert "secret" not in json.dumps(payload)


def _stub_meta_eval_sampling(monkeypatch, synthetic_arena_df: pd.DataFrame) -> None:
    monkeypatch.setattr(
        meta_eval_runner,
        "load_reference_arena_battles",
        lambda reference_arena, languages=None: synthetic_arena_df,
    )
    monkeypatch.setattr(
        meta_eval_runner,
        "select_top_models",
        lambda df, top_models: (["model-0", "model-1"], df),
    )
    monkeypatch.setattr(
        meta_eval_runner,
        "sample_battles_per_model",
        lambda df_top, models, battles_per_model, seed: df_top.head(1),
    )


def test_meta_eval_runner_creates_cache_cell_under_single_component_task(
    monkeypatch, tmp_path, synthetic_arena_df
):
    _stub_meta_eval_sampling(monkeypatch, synthetic_arena_df)
    args = _meta_args_with_cache(tmp_path, top_models=2, swap_mode="fixed")

    meta_eval_main(args)

    task_root = (
        Path(args.cache.store_root)
        / "inference"
        / meta_eval_cache_task(args.reference_arena)
    )
    db_files = list(task_root.rglob("inference.db"))
    assert db_files, f"expected cache cell under {task_root}"
    assert "meta-eval-LMArena-140k" in str(db_files[0])


def test_meta_eval_runner_skips_push_when_downstream_processing_fails(
    monkeypatch, tmp_path, synthetic_arena_df
):
    import judgearena.inference_cache as inference_cache_mod

    push_calls: list[tuple] = []
    monkeypatch.setattr(
        inference_cache_mod,
        "push_cells",
        lambda *args, **kwargs: push_calls.append((args, kwargs)),
    )
    _stub_meta_eval_sampling(monkeypatch, synthetic_arena_df)
    monkeypatch.setattr(
        meta_eval_runner,
        "_compute_results",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("downstream failed")),
    )

    args = _meta_args_with_cache(
        tmp_path,
        top_models=2,
        cache=CacheArgs(store_root=str(tmp_path / "cache"), cache_push=True),
    )

    with pytest.raises(RuntimeError, match="downstream failed"):
        meta_eval_main(args)

    assert push_calls == []
    task_root = (
        Path(args.cache.store_root)
        / "inference"
        / meta_eval_cache_task(args.reference_arena)
    )
    assert list(task_root.rglob("inference.db"))


@pytest.fixture
def synthetic_arena_df() -> pd.DataFrame:
    conv_a = [
        {"role": "user", "content": "Question 0"},
        {"role": "assistant", "content": "Answer A 0"},
    ]
    conv_b = [
        {"role": "user", "content": "Question 0"},
        {"role": "assistant", "content": "Answer B 0"},
    ]
    return pd.DataFrame(
        [
            {
                "question_id": "q-0",
                "tstamp": 1,
                "model_a": "model-0",
                "model_b": "model-1",
                "winner": "model_a",
                "conversation_a": conv_a,
                "conversation_b": conv_b,
                "benchmark": "LMArena-140k",
                "lang": "en",
            }
        ]
    )
