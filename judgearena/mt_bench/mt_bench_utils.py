"""MT-Bench evaluation pipeline.

Orchestrates multi-turn generation, FastChat-compatible pairwise judging,
and result saving for the MT-Bench benchmark.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.eval_utils import _compute_grouped_stats, print_results
from judgearena.generate import generate_multiturn
from judgearena.instruction_dataset import load_instructions
from judgearena.instruction_dataset.mt_bench import (
    load_mt_bench_model_answers,
    mt_bench_native_baseline,
)
from judgearena.log import get_logger
from judgearena.mt_bench.fastchat_compat import (
    FASTCHAT_TEMPERATURE_CONFIG,
    judge_mt_bench_pairwise_fastchat,
)
from judgearena.repro import _to_jsonable
from judgearena.utils import cache_function_dataframe, compute_pref_summary, make_model

logger = get_logger(__name__)

if TYPE_CHECKING:
    from judgearena.config import RunConfig


def _generate_mt_bench_completions(
    cfg: RunConfig,
    questions_df: pd.DataFrame,
    ignore_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_prefix = "mt-bench"

    def _run_generation(model_name: str) -> pd.DataFrame:
        return generate_multiturn(
            questions=questions_df,
            model=model_name,
            truncate_input_chars=cfg.generation.truncate_all_input_chars,
            max_tokens=cfg.model.max_out_tokens,
            use_tqdm=cfg.run.use_tqdm,
            max_model_len=cfg.model.max_model_len,
            chat_template=cfg.model.chat_template,
            temperature_config=FASTCHAT_TEMPERATURE_CONFIG,
            **cfg.model.engine_kwargs,
        )

    def _load_or_generate(model_name: str) -> pd.DataFrame:
        preloaded = load_mt_bench_model_answers(
            model_name, n_instructions=cfg.generation.n_instructions
        )
        if preloaded is not None:
            return preloaded.set_index("instruction_index").loc[questions_df.index]
        return (
            cache_function_dataframe(
                lambda: _run_generation(model_name),
                ignore_cache=ignore_cache,
                cache_name=f"{cache_prefix}_{model_name}_{cfg.generation.n_instructions}",
            )
            .set_index("instruction_index")
            .loc[questions_df.index]
        )

    return _load_or_generate(cfg.model.path), _load_or_generate(cfg.model.path_b)


def _build_mt_bench_result_name(cfg: RunConfig, suffix: str | None = None) -> str:
    name = f"{cfg.task}-{cfg.model.path}-{cfg.model.path_b}-{cfg.judge.model}"
    name += f"-{cfg.judge.swap_mode}"
    if suffix:
        name += f"-{suffix}"
    return name.replace("/", "_")


def _save_mt_bench_results(
    *,
    cfg: RunConfig,
    res_folder: Path,
    result_name: str,
    results: dict[str, object],
    annotations_df: pd.DataFrame,
) -> None:
    with open(res_folder / f"args-{result_name}.json", "w") as f:
        json.dump(_to_jsonable(cfg.model_dump()), f, indent=2, allow_nan=False)

    annotations_df.to_csv(res_folder / f"{result_name}-annotations.csv", index=False)

    with open(res_folder / f"results-{result_name}.json", "w") as f:
        json.dump(_to_jsonable(results), f, indent=2, allow_nan=False)


def _run_mt_bench_fastchat(
    *,
    cfg: RunConfig,
    res_folder: Path,
    result_name: str,
    questions_df: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    judge_chat_model,
) -> pd.Series:
    prefs, annotations, combined_metadata, num_inconsistent = (
        judge_mt_bench_pairwise_fastchat(
            judge_chat_model=judge_chat_model,
            judge_model=cfg.judge.model,
            questions=questions_df,
            completions_a=completions_a,
            completions_b=completions_b,
            model_a=cfg.model.path,
            model_b=cfg.model.path_b,
            turns_mode="both",
            swap_mode=cfg.judge.swap_mode,
            truncate_input_chars=cfg.generation.truncate_judge_input_chars,
            use_tqdm=cfg.run.use_tqdm,
        )
    )

    stats = compute_pref_summary(prefs)
    results = {
        "task": cfg.task,
        "model_A": cfg.model.path,
        "model_B": cfg.model.path_b,
        "judge_model": cfg.judge.model,
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
        cfg=cfg,
        res_folder=res_folder,
        result_name=result_name,
        results=results,
        annotations_df=pd.DataFrame(annotations),
    )
    return prefs


def run_mt_bench(
    cfg: RunConfig,
    ignore_cache: bool,
    *,
    res_folder: Path,
    result_name: str,
):
    """MT-Bench pipeline with FastChat-compatible pairwise judging."""
    if cfg.model.path_b is None:
        cfg.model.path_b = mt_bench_native_baseline(cfg.task)
    if cfg.model.path_b is None:
        raise ValueError(
            f"--model_B is required for dataset '{cfg.task}'; "
            "no dataset-native baseline registered."
        )
    questions_df = load_instructions("mt-bench", n_instructions=cfg.generation.n_instructions)
    logger.info(
        "Generating multi-turn completions for MT-Bench with %s and %s.",
        cfg.model.path,
        cfg.model.path_b,
    )
    completions_a, completions_b = _generate_mt_bench_completions(
        cfg=cfg,
        questions_df=questions_df,
        ignore_cache=ignore_cache,
    )
    judge_chat_model = make_model(
        model=cfg.judge.model,
        max_tokens=cfg.judge.max_out_tokens,
        temperature=0.0,
        max_model_len=cfg.judge.max_model_len,
        chat_template=cfg.model.chat_template,
        **{**cfg.model.engine_kwargs, **cfg.judge.engine_kwargs},
    )
    return _run_mt_bench_fastchat(
        cfg=cfg,
        res_folder=res_folder,
        result_name=result_name,
        questions_df=questions_df,
        completions_a=completions_a,
        completions_b=completions_b,
        judge_chat_model=judge_chat_model,
    )
