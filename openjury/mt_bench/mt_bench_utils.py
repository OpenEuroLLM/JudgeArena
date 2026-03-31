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

from openjury.eval_utils import _compute_grouped_stats, print_results
from openjury.generate import generate_multiturn
from openjury.instruction_dataset import load_instructions
from openjury.mt_bench.fastchat_compat import (
    FASTCHAT_TEMPERATURE_CONFIG,
    judge_mt_bench_pairwise_fastchat,
)
from openjury.repro import _to_jsonable
from openjury.utils import cache_function_dataframe, compute_pref_summary, make_model

if TYPE_CHECKING:
    from openjury.config import CliArgs

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
            turns_mode="both",
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
        temperature=0.0,
        max_model_len=args.max_model_len,
        chat_template=args.chat_template,
    )
    return _run_mt_bench_fastchat(
        args=args,
        questions_df=questions_df,
        completions_a=completions_a,
        completions_b=completions_b,
        judge_chat_model=judge_chat_model,
    )
