"""MT-Bench evaluation pipeline.

Orchestrates multi-turn generation, per-turn judging (OpenJury or
FastChat-compatible), and result saving for the MT-Bench benchmark.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from openjury.evaluate import PairScore, load_judge_system_and_user_prompt
from openjury.eval_utils import (
    _compute_grouped_stats,
    _make_judge_annotation,
    print_results,
)
from openjury.generate import generate_multiturn
from openjury.instruction_dataset import load_instructions
from openjury.mt_bench.common import iter_mt_bench_pairwise_rows
from openjury.mt_bench.fastchat_compat import (
    FASTCHAT_TEMPERATURE_CONFIG,
    judge_mt_bench_pairwise_fastchat,
)
from openjury.repro import _to_jsonable
from openjury.utils import cache_function_dataframe, compute_pref_summary, make_model

if TYPE_CHECKING:
    from openjury.config import CliArgs

NEED_REF_CATS = {"math", "reasoning", "coding"}


def format_mt_bench_for_evaluation(
    questions: pd.DataFrame,
    completions_A: pd.DataFrame,
    completions_B: pd.DataFrame,
    turns_mode: str,
    truncate_input_chars: int | None,
) -> tuple[
    tuple[list[str], list[str], list[str], list[dict[str, object]]],
    tuple[list[str], list[str], list[str], list[dict[str, object]]],
]:
    """Flatten MT-Bench into per-turn instruction/completion battle inputs."""
    assert turns_mode in ("both", "single", "multi")
    eval_single = turns_mode in ("both", "single")
    eval_multi = turns_mode in ("both", "multi")

    instructions_turn_1: list[str] = []
    completions_a_turn_1: list[str] = []
    completions_b_turn_1: list[str] = []
    metadata_turn_1: list[dict[str, object]] = []

    instructions_turn_2: list[str] = []
    completions_a_turn_2: list[str] = []
    completions_b_turn_2: list[str] = []
    metadata_turn_2: list[dict[str, object]] = []

    for row in iter_mt_bench_pairwise_rows(
        questions=questions,
        completions_a=completions_A,
        completions_b=completions_B,
        truncate_input_chars=truncate_input_chars,
    ):
        needs_ref = row.category in NEED_REF_CATS
        if eval_single:
            if needs_ref and row.ref_1:
                instruction = (
                    "[MT-Bench | Turn 1]\n"
                    "Use the reference answer for correctness checks.\n\n"
                    f"[Question]\n{row.turn_1_question}\n\n"
                    f"[Reference Answer]\n{row.ref_1}"
                )
            else:
                instruction = row.turn_1_question

            instructions_turn_1.append(instruction)
            completions_a_turn_1.append(row.answer_a_1)
            completions_b_turn_1.append(row.answer_b_1)
            metadata_turn_1.append(
                {
                    "question_id": row.question_id,
                    "category": row.category,
                    "turn": 1,
                }
            )

        if eval_multi and row.turn_2_question:
            instruction_parts = [
                "Please focus on which assistant provides a better answer to the second user question."
            ]
            if needs_ref and (row.ref_1 or row.ref_2):
                instruction_parts.extend(
                    [
                        "<|The Start of Reference Answer|>",
                        "### User:",
                        row.turn_1_question,
                        "### Reference answer:",
                        row.ref_1,
                        "### User:",
                        row.turn_2_question,
                        "### Reference answer:",
                        row.ref_2,
                        "<|The End of Reference Answer|>",
                    ]
                )

            conversation_a = _format_mt_bench_multiturn_conversation(
                turn_1_question=row.turn_1_question,
                turn_1_answer=row.answer_a_1,
                turn_2_question=row.turn_2_question,
                turn_2_answer=row.answer_a_2,
            )
            conversation_b = _format_mt_bench_multiturn_conversation(
                turn_1_question=row.turn_1_question,
                turn_1_answer=row.answer_b_1,
                turn_2_question=row.turn_2_question,
                turn_2_answer=row.answer_b_2,
            )

            instructions_turn_2.append("\n\n".join(instruction_parts))
            completions_a_turn_2.append(conversation_a)
            completions_b_turn_2.append(conversation_b)
            metadata_turn_2.append(
                {
                    "question_id": row.question_id,
                    "category": row.category,
                    "turn": 2,
                }
            )

    return (
        (
            instructions_turn_1,
            completions_a_turn_1,
            completions_b_turn_1,
            metadata_turn_1,
        ),
        (
            instructions_turn_2,
            completions_a_turn_2,
            completions_b_turn_2,
            metadata_turn_2,
        ),
    )


def _format_mt_bench_multiturn_conversation(
    *,
    turn_1_question: str,
    turn_1_answer: str,
    turn_2_question: str,
    turn_2_answer: str,
) -> str:
    return (
        "### User:\n"
        f"{turn_1_question}\n\n"
        "### Assistant:\n"
        f"{turn_1_answer}\n\n"
        "### User:\n"
        f"{turn_2_question}\n\n"
        "### Assistant:\n"
        f"{turn_2_answer}"
    )


def _generate_mt_bench_completions(
    args: CliArgs,
    questions_df: pd.DataFrame,
    ignore_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_prefix = (
        "mt-bench_fastchatgen" if args.mt_bench_compatibility == "fastchat" else "mt-bench"
    )

    def _run_generation(model_name: str) -> pd.DataFrame:
        if args.mt_bench_compatibility == "fastchat":
            return generate_multiturn(
                questions=questions_df,
                model=model_name,
                truncate_input_chars=args.truncate_all_input_chars,
                max_tokens=args.max_out_tokens_models,
                use_tqdm=args.use_tqdm,
                max_model_len=args.max_model_len,
                chat_template=args.chat_template,
                temperature_config=FASTCHAT_TEMPERATURE_CONFIG,
            )
        return generate_multiturn(
            questions=questions_df,
            model=model_name,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            use_tqdm=args.use_tqdm,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
        )

    completions_a = cache_function_dataframe(
        lambda: _run_generation(args.model_A),
        ignore_cache=ignore_cache,
        cache_name=f"{cache_prefix}_{args.model_A}_{args.n_instructions}",
    ).set_index("instruction_index")

    completions_b = cache_function_dataframe(
        lambda: _run_generation(args.model_B),
        ignore_cache=ignore_cache,
        cache_name=f"{cache_prefix}_{args.model_B}_{args.n_instructions}",
    ).set_index("instruction_index")
    return completions_a, completions_b


def _build_mt_bench_result_name(args: CliArgs, suffix: str | None = None) -> str:
    name = f"{args.dataset}-{args.model_A}-{args.model_B}-{args.judge_model}"
    name += f"-{args.swap_mode}"
    if suffix:
        name += f"-{suffix}"
    return name.replace("/", "_")


def _save_mt_bench_results(
    *,
    args: CliArgs,
    results: dict[str, object],
    annotations_df: pd.DataFrame,
    name_suffix: str | None = None,
) -> None:
    name = _build_mt_bench_result_name(args, suffix=name_suffix)
    res_folder = Path(args.result_folder) / name
    res_folder.mkdir(parents=True, exist_ok=True)

    with open(res_folder / f"args-{name}.json", "w") as f:
        json.dump(_to_jsonable(asdict(args)), f, indent=2, allow_nan=False)

    annotations_df.to_csv(res_folder / f"{name}-annotations.csv", index=False)

    with open(res_folder / f"results-{name}.json", "w") as f:
        json.dump(_to_jsonable(results), f, indent=2, allow_nan=False)


def _run_mt_bench_fastchat(
    *,
    args: CliArgs,
    questions_df: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    judge_chat_model,
) -> pd.Series:
    prefs, annotations, combined_metadata, num_inconsistent = (
        judge_mt_bench_pairwise_fastchat(
            judge_chat_model=judge_chat_model,
            judge_model=args.judge_model,
            questions=questions_df,
            completions_a=completions_a,
            completions_b=completions_b,
            model_a=args.model_A,
            model_b=args.model_B,
            turns_mode=args.mt_bench_turns,
            swap_mode=args.swap_mode,
            truncate_input_chars=args.truncate_all_input_chars,
            use_tqdm=args.use_tqdm,
        )
    )

    stats = compute_pref_summary(prefs)
    results = {
        "dataset": args.dataset,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
        "mt_bench_compatibility": args.mt_bench_compatibility,
        "num_inconsistent": num_inconsistent,
        **stats,
        "per_category": _compute_grouped_stats(prefs, combined_metadata, "category"),
        "per_turn": _compute_grouped_stats(prefs, combined_metadata, "turn"),
        "preferences": prefs.tolist(),
        "date": str(datetime.now().isoformat()),
        "user": os.getenv("USER", ""),
    }
    print_results(results)
    _save_mt_bench_results(
        args=args,
        results=results,
        annotations_df=pd.DataFrame(annotations),
        name_suffix=f"mtbench_{args.mt_bench_compatibility}",
    )
    return prefs


def _run_mt_bench_openjury(
    *,
    args: CliArgs,
    questions_df: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    judge_chat_model,
) -> pd.Series:
    turn_1_inputs, turn_2_inputs = format_mt_bench_for_evaluation(
        questions=questions_df,
        completions_A=completions_a,
        completions_B=completions_b,
        turns_mode=args.mt_bench_turns,
        truncate_input_chars=args.truncate_all_input_chars,
    )
    (
        instructions_turn_1,
        completions_a_turn_1,
        completions_b_turn_1,
        metadata_turn_1,
    ) = turn_1_inputs
    (
        instructions_turn_2,
        completions_a_turn_2,
        completions_b_turn_2,
        metadata_turn_2,
    ) = turn_2_inputs

    score_parser = PairScore()
    annotations = []
    metadata_for_annotations: list[dict[str, object]] = []
    annotations_reversed = []
    metadata_for_reversed_annotations: list[dict[str, object]] = []
    preference_parts: list[pd.Series] = []
    combined_metadata: list[dict[str, object]] = []

    if instructions_turn_1:
        turn_1 = _make_judge_annotation(
            judge_chat_model=judge_chat_model,
            instructions=instructions_turn_1,
            completions_A=completions_a_turn_1,
            completions_B=completions_b_turn_1,
            metadata=metadata_turn_1,
            score_parser=score_parser,
            provide_explanation=args.provide_explanation,
            swap_mode=args.swap_mode,
            truncate_input_chars=args.truncate_all_input_chars,
            use_tqdm=args.use_tqdm,
        )
        annotations.extend(turn_1.annotations)
        annotations_reversed.extend(turn_1.annotations_reversed)
        metadata_for_annotations.extend(turn_1.metadata_for_annotations)
        metadata_for_reversed_annotations.extend(
            turn_1.metadata_for_reversed_annotations
        )
        preference_parts.append(turn_1.preferences)
        combined_metadata.extend(turn_1.combined_metadata)

    if instructions_turn_2:
        mt_system_prompt, mt_user_prompt_template = load_judge_system_and_user_prompt(
            provide_explanation=args.provide_explanation,
            multi_turn=True,
        )
        turn_2 = _make_judge_annotation(
            judge_chat_model=judge_chat_model,
            instructions=instructions_turn_2,
            completions_A=completions_a_turn_2,
            completions_B=completions_b_turn_2,
            metadata=metadata_turn_2,
            score_parser=score_parser,
            provide_explanation=args.provide_explanation,
            swap_mode=args.swap_mode,
            truncate_input_chars=args.truncate_all_input_chars,
            use_tqdm=args.use_tqdm,
            system_prompt=mt_system_prompt,
            user_prompt_template=mt_user_prompt_template,
        )
        annotations.extend(turn_2.annotations)
        annotations_reversed.extend(turn_2.annotations_reversed)
        metadata_for_annotations.extend(turn_2.metadata_for_annotations)
        metadata_for_reversed_annotations.extend(
            turn_2.metadata_for_reversed_annotations
        )
        preference_parts.append(turn_2.preferences)
        combined_metadata.extend(turn_2.combined_metadata)

    prefs = (
        pd.concat(preference_parts).reset_index(drop=True)
        if preference_parts
        else pd.Series(dtype=float)
    )
    stats = compute_pref_summary(prefs)
    results = {
        "dataset": args.dataset,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
        **stats,
        "per_category": _compute_grouped_stats(prefs, combined_metadata, "category"),
        "per_turn": _compute_grouped_stats(prefs, combined_metadata, "turn"),
        "preferences": prefs.tolist(),
        "date": str(datetime.now().isoformat()),
        "user": os.getenv("USER", ""),
    }
    print_results(results)

    df = pd.DataFrame(annotations)
    df["instruction_index"] = [meta["question_id"] for meta in metadata_for_annotations]
    df["category"] = [meta["category"] for meta in metadata_for_annotations]
    df["turn"] = [meta["turn"] for meta in metadata_for_annotations]
    df["model_A"] = args.model_A
    df["model_B"] = args.model_B
    df["judge"] = args.judge_model

    if args.swap_mode == "both":
        df_reversed = pd.DataFrame(annotations_reversed)
        df_reversed["instruction_index"] = [
            meta["question_id"] for meta in metadata_for_reversed_annotations
        ]
        df_reversed["category"] = [
            meta["category"] for meta in metadata_for_reversed_annotations
        ]
        df_reversed["turn"] = [meta["turn"] for meta in metadata_for_reversed_annotations]
        df_reversed["model_A"] = args.model_B
        df_reversed["model_B"] = args.model_A
        df_reversed["judge"] = args.judge_model
        df = pd.concat([df, df_reversed], ignore_index=True)

    _save_mt_bench_results(
        args=args,
        results=results,
        annotations_df=df,
    )
    return prefs


def run_mt_bench(args: CliArgs, ignore_cache: bool):
    """MT-Bench pipeline (optionally FastChat-compatible)."""
    questions_df = load_instructions("mt-bench", n_instructions=args.n_instructions)
    print(
        f"Generating multi-turn completions for MT-Bench with {args.model_A} and {args.model_B}."
    )
    completions_a, completions_b = _generate_mt_bench_completions(
        args=args,
        questions_df=questions_df,
        ignore_cache=ignore_cache,
    )
    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=args.max_out_tokens_judge,
        temperature=0.0 if args.mt_bench_compatibility == "fastchat" else None,
        max_model_len=args.max_model_len,
        chat_template=args.chat_template,
    )
    if args.mt_bench_compatibility == "fastchat":
        return _run_mt_bench_fastchat(
            args=args,
            questions_df=questions_df,
            completions_a=completions_a,
            completions_b=completions_b,
            judge_chat_model=judge_chat_model,
        )
    return _run_mt_bench_openjury(
        args=args,
        questions_df=questions_df,
        completions_a=completions_a,
        completions_b=completions_b,
        judge_chat_model=judge_chat_model,
    )
