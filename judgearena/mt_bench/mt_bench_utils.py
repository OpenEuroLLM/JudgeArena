"""MT-Bench evaluation pipeline.

Orchestrates multi-turn generation, FastChat-compatible pairwise judging,
and result saving for the MT-Bench benchmark.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.eval_utils import _compute_grouped_stats, print_results
from judgearena.generate import generate_multiturn
from judgearena.instruction_dataset import load_instructions
from judgearena.instruction_dataset.mt_bench import load_mt_bench_model_answers
from judgearena.mt_bench.fastchat_compat import (
    FASTCHAT_TEMPERATURE_CONFIG,
    judge_mt_bench_pairwise_fastchat,
)
from judgearena.repro import _to_jsonable
from judgearena.utils import cache_function_dataframe, compute_pref_summary, make_model

if TYPE_CHECKING:
    from judgearena.generate_and_evaluate import CliArgs


# Use distinct first tokens for constrained decoding. The shared `[[` prefix
# caused the MT-Bench judge to collapse to `[[A]]` on every comparison.
_MIN_MT_BENCH_JUDGE_TOKENS = 2048
_DEFAULT_MT_BENCH_JUDGE_THINKING_TOKEN_BUDGET = 192
_MT_BENCH_REASONING_MAX_CHARS = 384


def build_mt_bench_verdict_json_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "maxLength": _MT_BENCH_REASONING_MAX_CHARS,
            },
            "verdict": {"type": "string", "enum": ["A", "B", "C"]},
        },
        "required": ["reasoning", "verdict"],
        "additionalProperties": False,
    }


def _align_mt_bench_completions(
    *, questions_df: pd.DataFrame, completions: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    indexed = completions.set_index("instruction_index")
    missing_ids = questions_df.index.difference(indexed.index)
    if not missing_ids.empty:
        missing_ids_preview = ", ".join(str(x) for x in missing_ids[:5])
        raise ValueError(
            f"MT-Bench completions for '{model_name}' are missing "
            f"{len(missing_ids)} question(s). First missing ids: {missing_ids_preview}."
        )
    return indexed.loc[questions_df.index]


def _generate_mt_bench_completions(
    args: CliArgs,
    questions_df: pd.DataFrame,
    ignore_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_prefix = "mt-bench"

    def _run_generation(model_name: str) -> pd.DataFrame:
        return generate_multiturn(
            questions=questions_df,
            model=model_name,
            truncate_input_chars=args.truncate_all_input_chars,
            max_tokens=args.max_out_tokens_models,
            use_tqdm=args.use_tqdm,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            temperature_config=FASTCHAT_TEMPERATURE_CONFIG,
            **args.engine_kwargs,
        )

    def _load_or_generate(model_name: str) -> pd.DataFrame:
        loaded_answers = load_mt_bench_model_answers(
            model_name, n_instructions=args.n_instructions
        )
        if loaded_answers is not None:
            print(f"Using pre-generated MT-Bench answers for '{model_name}'.")
            return _align_mt_bench_completions(
                questions_df=questions_df,
                completions=loaded_answers,
                model_name=model_name,
            )
        generated_answers = cache_function_dataframe(
            lambda: _run_generation(model_name),
            ignore_cache=ignore_cache,
            cache_name=f"{cache_prefix}_{model_name}_{args.n_instructions}",
        )
        return _align_mt_bench_completions(
            questions_df=questions_df,
            completions=generated_answers,
            model_name=model_name,
        )

    completions_a = _load_or_generate(args.model_A)
    completions_b = _load_or_generate(args.model_B)
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
    constrained_plain_verdict: bool,
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
            turns_mode="both",
            swap_mode=args.swap_mode,
            truncate_input_chars=args.truncate_all_input_chars,
            use_tqdm=args.use_tqdm,
            constrained_plain_verdict=constrained_plain_verdict,
        )
    )

    stats = compute_pref_summary(prefs)
    results = {
        "dataset": args.dataset,
        "model_A": args.model_A,
        "model_B": args.model_B,
        "judge_model": args.judge_model,
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
        name_suffix="mtbench",
    )
    return prefs


def run_mt_bench(args: CliArgs, ignore_cache: bool):
    """MT-Bench pipeline with FastChat-compatible pairwise judging."""
    if args.swap_mode != "both":
        print(
            "MT-Bench requires swap_mode='both' to match FastChat and correct "
            f"for position bias; overriding requested swap_mode='{args.swap_mode}'."
        )
        args.swap_mode = "both"
    if args.max_out_tokens_judge < _MIN_MT_BENCH_JUDGE_TOKENS:
        print(
            "MT-Bench judge prompts require room for explanation plus verdict; "
            f"overriding max_out_tokens_judge from {args.max_out_tokens_judge} "
            f"to {_MIN_MT_BENCH_JUDGE_TOKENS}."
        )
        args.max_out_tokens_judge = _MIN_MT_BENCH_JUDGE_TOKENS
    questions_df = load_instructions("mt-bench", n_instructions=args.n_instructions)
    print(
        f"Generating multi-turn completions for MT-Bench with {args.model_A} and {args.model_B}."
    )
    completions_a, completions_b = _generate_mt_bench_completions(
        args=args,
        questions_df=questions_df,
        ignore_cache=ignore_cache,
    )
    judge_model_kwargs = dict(args.engine_kwargs)
    judge_model_kwargs["structured_outputs_json"] = build_mt_bench_verdict_json_schema()
    judge_model_kwargs.setdefault(
        "thinking_token_budget", _DEFAULT_MT_BENCH_JUDGE_THINKING_TOKEN_BUDGET
    )

    judge_chat_model = make_model(
        model=args.judge_model,
        max_tokens=args.max_out_tokens_judge,
        temperature=0.0,
        max_model_len=args.max_model_len,
        chat_template=args.chat_template,
        **judge_model_kwargs,
    )
    return _run_mt_bench_fastchat(
        args=args,
        questions_df=questions_df,
        completions_a=completions_a,
        completions_b=completions_b,
        judge_chat_model=judge_chat_model,
        constrained_plain_verdict=False,
    )
