"""Run LLM judge annotations for meta-evaluation battles."""

from __future__ import annotations

import pandas as pd

from judgearena.arenas_utils import extract_turn_text
from judgearena.evaluate import JudgeAnnotation, annotate_battles
from judgearena.meta_eval.cache import (
    AnnotationCache,
    AnnotationEntry,
    AnnotationKey,
)
from judgearena.meta_eval.cli_args import CliMetaEvalArgs
from judgearena.meta_eval.cost import (
    estimate_annotation_cost_usd,
    estimate_token_count,
)
from judgearena.meta_eval.parsers import add_parsed_columns, invert_winner


def _battle_texts(df_batch: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    instructions = [extract_turn_text(conv[0]) for conv in df_batch["conversation_a"]]
    completions_a = [
        extract_turn_text(conv[1]) if len(conv) > 1 else ""
        for conv in df_batch["conversation_a"]
    ]
    completions_b = [
        extract_turn_text(conv[1]) if len(conv) > 1 else ""
        for conv in df_batch["conversation_b"]
    ]
    return instructions, completions_a, completions_b


def _swap_batch(df_batch: pd.DataFrame) -> pd.DataFrame:
    swapped = df_batch.copy()
    swapped["conversation_a"] = df_batch["conversation_b"]
    swapped["conversation_b"] = df_batch["conversation_a"]
    swapped["model_a"] = df_batch["model_b"]
    swapped["model_b"] = df_batch["model_a"]
    return swapped


def _annotations_to_frame(
    df_batch: pd.DataFrame,
    annotations,
    *,
    prompt_mode: str,
    judge_model: str,
) -> pd.DataFrame:
    rows = []
    for ann, (_, battle) in zip(annotations, df_batch.iterrows(), strict=True):
        judge_input = ann.judge_input or ""
        cost_usd, cost_source = estimate_annotation_cost_usd(
            judge_input=judge_input,
            judge_completion=ann.judge_completion,
            judge_model=judge_model,
        )
        rows.append(
            {
                "question_id": battle["question_id"],
                "model_a": battle["model_a"],
                "model_b": battle["model_b"],
                "winner": battle["winner"],
                "lang": battle["lang"],
                "benchmark": battle["benchmark"],
                "instruction": ann.instruction,
                "completion_a": ann.completion_A,
                "completion_b": ann.completion_B,
                "judge_input": ann.judge_input,
                "judge_completion": ann.judge_completion,
                "estimated_input_tokens": estimate_token_count(judge_input),
                "estimated_output_tokens": estimate_token_count(ann.judge_completion),
                "cost_usd": cost_usd,
                "cost_source": cost_source,
            }
        )
    return add_parsed_columns(pd.DataFrame(rows), prompt_mode)


def _judge_cache_name(args: CliMetaEvalArgs) -> str:
    if args.prompt_mode == "standard":
        return args.judge_model
    return f"{args.judge_model}::{args.prompt_mode}"


def _cache_keys(
    df_batch: pd.DataFrame,
    *,
    judge: str,
) -> list[AnnotationKey]:
    return [
        AnnotationKey(
            benchmark=str(battle["benchmark"]),
            instruction_id=str(battle["question_id"]),
            model_a=str(battle["model_a"]),
            model_b=str(battle["model_b"]),
            judge=judge,
        )
        for _, battle in df_batch.iterrows()
    ]


def _annotation_from_entry(
    entry: AnnotationEntry,
    *,
    instruction: str,
    completion_a: str,
    completion_b: str,
) -> JudgeAnnotation:
    return JudgeAnnotation(
        instruction=instruction,
        completion_A=completion_a,
        completion_B=completion_b,
        judge_completion=entry.judge_completion,
        judge_input=entry.judge_input,
    )


def _run_cached_batch(
    df_batch: pd.DataFrame,
    args: CliMetaEvalArgs,
    *,
    judge_chat_model,
    annotation_cache: AnnotationCache,
    prompt_spec,
    swapped: bool,
) -> pd.DataFrame:
    working = _swap_batch(df_batch) if swapped else df_batch
    instructions, completions_a, completions_b = _battle_texts(working)
    judge = _judge_cache_name(args)
    keys = _cache_keys(working, judge=judge)
    cached_entries = (
        [None] * len(keys)
        if args.ignore_cache
        else annotation_cache.batch_get_annotations(keys)
    )
    missing_indices = [
        index for index, entry in enumerate(cached_entries) if entry is None
    ]

    if missing_indices:
        new_annotations = annotate_battles(
            judge_chat_model=judge_chat_model,
            instructions=[instructions[index] for index in missing_indices],
            completions_A=[completions_a[index] for index in missing_indices],
            completions_B=[completions_b[index] for index in missing_indices],
            system_prompt=prompt_spec.system_prompt,
            user_prompt_template=prompt_spec.user_prompt_template,
            truncate_input_chars=args.truncate_judge_input_chars,
            provide_explanation=args.provide_explanation,
        )
        new_entries = [
            AnnotationEntry(
                **key.__dict__,
                judge_input=annotation.judge_input or "",
                judge_completion=annotation.judge_completion,
            )
            for key, annotation in zip(
                [keys[index] for index in missing_indices],
                new_annotations,
                strict=True,
            )
        ]
        annotation_cache.batch_put(new_entries)
        for index, entry in zip(missing_indices, new_entries, strict=True):
            cached_entries[index] = entry

    annotations = [
        _annotation_from_entry(
            entry,
            instruction=instruction,
            completion_a=completion_a,
            completion_b=completion_b,
        )
        for entry, instruction, completion_a, completion_b in zip(
            cached_entries,
            instructions,
            completions_a,
            completions_b,
            strict=True,
        )
        if entry is not None
    ]
    return _annotations_to_frame(
        working,
        annotations,
        prompt_mode=args.prompt_mode,
        judge_model=args.judge_model,
    )


def _normalize_pass_frame(
    pass_frame: pd.DataFrame,
    original_batch: pd.DataFrame,
    *,
    orientation: str,
) -> pd.DataFrame:
    normalized = pass_frame.copy()
    normalized["orientation"] = orientation
    normalized["presented_model_a"] = pass_frame["model_a"].tolist()
    normalized["presented_model_b"] = pass_frame["model_b"].tolist()
    normalized["presented_completion_a"] = pass_frame["completion_a"].tolist()
    normalized["presented_completion_b"] = pass_frame["completion_b"].tolist()
    normalized["model_a"] = original_batch["model_a"].tolist()
    normalized["model_b"] = original_batch["model_b"].tolist()
    normalized["winner"] = original_batch["winner"].tolist()
    if orientation == "swapped":
        normalized["completion_a"] = pass_frame["completion_b"].tolist()
        normalized["completion_b"] = pass_frame["completion_a"].tolist()
        normalized["winner_llm"] = [
            invert_winner(winner) for winner in pass_frame["winner_llm"]
        ]
        normalized["pref_llm"] = 1.0 - pass_frame["pref_llm"]
    return normalized


def annotate_sample(
    df_sample: pd.DataFrame,
    args: CliMetaEvalArgs,
    *,
    judge_chat_model,
    prompt_spec,
    annotation_cache: AnnotationCache | None = None,
) -> pd.DataFrame:
    n_total = len(df_sample)
    n_batches = (n_total + args.batch_size - 1) // args.batch_size
    parts: list[pd.DataFrame] = []
    owns_cache = annotation_cache is None
    cache = annotation_cache or AnnotationCache()

    try:
        for batch_idx in range(n_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, n_total)
            df_batch = df_sample.iloc[start:end].copy()
            batch_df = _run_cached_batch(
                df_batch,
                args,
                judge_chat_model=judge_chat_model,
                annotation_cache=cache,
                prompt_spec=prompt_spec,
                swapped=False,
            )
            parts.append(
                _normalize_pass_frame(
                    batch_df,
                    df_batch,
                    orientation="forward",
                )
            )

            if args.swap_mode == "both":
                swapped_df = _run_cached_batch(
                    df_batch,
                    args,
                    judge_chat_model=judge_chat_model,
                    annotation_cache=cache,
                    prompt_spec=prompt_spec,
                    swapped=True,
                )
                parts.append(
                    _normalize_pass_frame(
                        swapped_df,
                        df_batch,
                        orientation="swapped",
                    )
                )
    finally:
        if owns_cache:
            cache.close()

    return pd.concat(parts, ignore_index=True)
