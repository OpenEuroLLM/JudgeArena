"""Run LLM judge annotations for meta-evaluation battles."""

from __future__ import annotations

from typing import Any

import pandas as pd

from judgearena.arenas_utils import extract_turn_text
from judgearena.evaluate import JudgeAnnotation, annotate_battles
from judgearena.inference_cache import InferenceCache
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
    annotations: list[JudgeAnnotation],
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


def _row_metadata(
    df_batch: pd.DataFrame,
    args: CliMetaEvalArgs,
    *,
    orientation: str,
) -> list[dict[str, Any]]:
    return [
        {
            "reference_arena": args.reference_arena,
            "benchmark": str(battle["benchmark"]),
            "question_id": str(battle["question_id"]),
            "presented_model_a": str(battle["model_a"]),
            "presented_model_b": str(battle["model_b"]),
            "prompt_mode": args.prompt_mode,
            "orientation": orientation,
        }
        for _, battle in df_batch.iterrows()
    ]


def _run_batch(
    df_batch: pd.DataFrame,
    args: CliMetaEvalArgs,
    *,
    judge_chat_model,
    prompt_spec,
    cache: InferenceCache | None,
    swapped: bool,
) -> pd.DataFrame:
    working = _swap_batch(df_batch) if swapped else df_batch
    instructions, completions_a, completions_b = _battle_texts(working)
    orientation = "swapped" if swapped else "forward"
    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions,
        completions_A=completions_a,
        completions_B=completions_b,
        system_prompt=prompt_spec.system_prompt,
        user_prompt_template=prompt_spec.user_prompt_template,
        truncate_input_chars=args.truncate_judge_input_chars,
        provide_explanation=args.provide_explanation,
        strip_thinking_before_judging=args.strip_thinking_before_judging,
        cache=cache,
        row_metadata=_row_metadata(working, args, orientation=orientation),
    )
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
    cache: InferenceCache | None = None,
) -> pd.DataFrame:
    n_total = len(df_sample)
    n_batches = (n_total + args.batch_size - 1) // args.batch_size
    parts: list[pd.DataFrame] = []

    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, n_total)
        df_batch = df_sample.iloc[start:end].copy()
        batch_df = _run_batch(
            df_batch,
            args,
            judge_chat_model=judge_chat_model,
            prompt_spec=prompt_spec,
            cache=cache,
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
            swapped_df = _run_batch(
                df_batch,
                args,
                judge_chat_model=judge_chat_model,
                prompt_spec=prompt_spec,
                cache=cache,
                swapped=True,
            )
            parts.append(
                _normalize_pass_frame(
                    swapped_df,
                    df_batch,
                    orientation="swapped",
                )
            )

    return pd.concat(parts, ignore_index=True)
