"""Registered MT-Bench-101 runner and evaluation pipeline."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.artifacts import prepare_run_directory, write_run_metadata_safely
from judgearena.benchmarks.mt_bench_101.evaluate import (
    derive_mt_bench_101_pairwise_preferences,
    judge_mt_bench_101_single,
    summarize_mt_bench_101_absolute_scores,
)
from judgearena.benchmarks.mt_bench_101.generate import (
    generate_mt_bench_101_completions,
)
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.datasets import load_instructions
from judgearena.log import get_logger
from judgearena.models import make_model
from judgearena.reports import BattleReport
from judgearena.tasks.schema import MTBench101Protocol
from judgearena.utils import cache_function_dataframe, generation_cache_token

logger = get_logger(__name__)

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.tasks.schema import ResolvedTaskSpec


def _select_complete_dialogues(
    eval_items: pd.DataFrame, n_instructions: int | None
) -> pd.DataFrame:
    if n_instructions is None:
        return eval_items
    selected = eval_items["dialogue_uid"].drop_duplicates().head(n_instructions)
    return eval_items.loc[eval_items["dialogue_uid"].isin(selected)]


def _generate_cached(
    *,
    cfg: RunConfig,
    eval_items: pd.DataFrame,
    model_name: str,
    role: str,
) -> pd.DataFrame:
    if role == "A":
        generation_kwargs = cfg.model.evaluated_generation_kwargs()
    elif role == "B":
        generation_kwargs = cfg.model.baseline_generation_kwargs()
    else:
        raise ValueError(f"Unknown generation role: {role!r}")
    sampling_token = generation_cache_token(generation_kwargs)
    return cache_function_dataframe(
        lambda: generate_mt_bench_101_completions(
            eval_items=eval_items,
            model=model_name,
            truncate_input_chars=cfg.generation.truncate_all_input_chars,
            use_tqdm=cfg.run.use_tqdm,
            **generation_kwargs,
        ),
        ignore_cache=cfg.run.ignore_cache,
        cache_name=(
            f"{cfg.task}_{model_name}_{cfg.generation.n_instructions}_{sampling_token}"
        ),
    )


def run_mt_bench_101_benchmark(
    cfg: RunConfig, task: ResolvedTaskSpec | None = None
) -> pd.Series:
    """Generate golden-context answers, grade each turn, then derive pairwise prefs."""
    run_started_at = datetime.now(UTC)
    protocol = task.spec.protocol if task is not None else None
    if not isinstance(protocol, MTBench101Protocol):
        raise ValueError(f"Task {cfg.task!r} does not define an MT-Bench-101 protocol.")
    if cfg.model.baseline is None:
        raise ValueError(
            f"model.baseline is required for task {cfg.task!r} "
            "(runtime_required baseline)."
        )
    result_name = (
        f"{cfg.task}-{cfg.model.name}-{cfg.model.baseline}-{cfg.judge.model}"
    ).replace("/", "_")
    res_folder = prepare_run_directory(
        cfg,
        Path(cfg.run.result_folder)
        / f"{result_name}-{run_started_at.strftime('%Y%m%d_%H%M%S')}",
    )
    eval_items = _select_complete_dialogues(
        load_instructions(task if task is not None else cfg.task, n_instructions=None),
        cfg.generation.n_instructions,
    )
    logger.info(
        "Generating golden-context completions for MT-Bench-101 with %s and %s.",
        cfg.model.name,
        cfg.model.baseline,
    )
    completions_a = _generate_cached(
        cfg=cfg, eval_items=eval_items, model_name=cfg.model.name, role="A"
    )
    completions_b = _generate_cached(
        cfg=cfg, eval_items=eval_items, model_name=cfg.model.baseline, role="B"
    )
    judge_model_kwargs = cfg.judge.model_kwargs(
        base_engine_kwargs=cfg.model.engine_kwargs,
        fallback_chat_template=cfg.model.chat_template,
    )
    if cfg.judge.temperature is None:
        judge_model_kwargs.setdefault("temperature", protocol.judge.default_temperature)
    if protocol.judge.default_max_out_tokens is not None:
        judge_model_kwargs.setdefault(
            "max_tokens", protocol.judge.default_max_out_tokens
        )
    judge_chat_model = make_model(model=cfg.judge.model, **judge_model_kwargs)
    scored_a = judge_mt_bench_101_single(
        judge_chat_model=judge_chat_model,
        eval_items=eval_items,
        completions=completions_a,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        use_tqdm=cfg.run.use_tqdm,
    )
    scored_b = judge_mt_bench_101_single(
        judge_chat_model=judge_chat_model,
        eval_items=eval_items,
        completions=completions_b,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        use_tqdm=cfg.run.use_tqdm,
    )
    pairwise = derive_mt_bench_101_pairwise_preferences(scored_a, scored_b)
    battles = pd.DataFrame(
        {
            "instruction_index": pairwise["instruction_index"],
            "task": pairwise["task"],
            "ability": pairwise["ability"],
            "dialogue_uid": pairwise["dialogue_uid"],
            "turn": pairwise["turn_index"],
            "model_a": cfg.model.name,
            "model_b": cfg.model.baseline,
            "evaluation_model": cfg.model.name,
            "pref": pairwise["preference"],
            "source": "llm-judge",
        }
    )
    metrics = build_metrics(protocol.scoring.metrics)
    metric_results = calculate_metrics(battles, metrics)
    report = BattleReport(
        task=cfg.task,
        model_a=cfg.model.name,
        model_b=cfg.model.baseline,
        judge_model=cfg.judge.model,
        metrics=metric_results,
        preferences=pairwise["preference"].tolist(),
        metadata={
            "evaluation_mode": "single_answer_grading",
            "judge_temperature": judge_model_kwargs.get("temperature"),
            "model_A_scores": summarize_mt_bench_101_absolute_scores(scored_a),
            "model_B_scores": summarize_mt_bench_101_absolute_scores(scored_b),
            "date": datetime.now(UTC).isoformat(),
            "user": os.getenv("USER", ""),
        },
        result_folder=str(res_folder),
    )
    report.render()
    report.save(res_folder / f"results-{result_name}.json")
    annotations = pd.concat(
        [
            scored_a.assign(evaluated_model=cfg.model.name),
            scored_b.assign(evaluated_model=cfg.model.baseline),
        ],
        ignore_index=True,
    )
    annotations.to_csv(res_folder / f"{result_name}-annotations.csv", index=False)
    write_run_metadata_safely(
        output_dir=res_folder,
        entrypoint="judgearena.benchmarks.mt_bench_101.runner.run_mt_bench_101_benchmark",
        run=cfg.model_dump(),
        results=report.to_dict(),
        input_payloads={
            "instruction_index": eval_items.index.tolist(),
            "dialogue_uid": eval_items["dialogue_uid"].tolist(),
        },
        started_at_utc=run_started_at,
    )
    return pd.Series(pairwise["preference"].tolist())
