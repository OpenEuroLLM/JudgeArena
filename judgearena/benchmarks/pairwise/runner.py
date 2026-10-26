"""
This script generates completions for a given task (dataset) and model,
and then evaluates them using a judge model.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from judgearena.artifacts import (
    prepare_run_directory,
    run_logging,
    write_annotations,
    write_run_metadata_safely,
)
from judgearena.battles import judged_battles
from judgearena.benchmarks.execution import build_judge
from judgearena.benchmarks.pairwise.baselines import resolve_baseline_plan
from judgearena.benchmarks.pairwise.comparisons import prepare_comparisons
from judgearena.benchmarks.pairwise.judging import prepare_pairwise_judging
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.datasets import load_task_data, resolve_dataset_filters
from judgearena.evaluate import resolve_run_judge_prompt
from judgearena.log import get_logger
from judgearena.reports import BattleReport
from judgearena.tasks.schema import ResolvedTaskSpec

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


@run_logging()
def run_pairwise(cfg: "RunConfig", resolved_task: ResolvedTaskSpec | None = None):
    """Load or generate candidates, judge pairs, and save annotations and metrics."""

    run_started_at = datetime.now(UTC)

    cfg = cfg.for_task(resolved_task)
    resolved_task = cfg.resolved_task
    resolved_prompt = resolve_run_judge_prompt(
        resolved_task,
        cfg.judge,
        multi_turn=resolved_task.spec.protocol.generation.mode == "multi_turn_chat",
    )
    metrics = build_metrics(resolved_task.spec.protocol.scoring.metrics)
    task_data = load_task_data(resolved_task).select(
        resolve_dataset_filters(resolved_task, cfg.run.filters)
    )
    instructions_df = task_data.instructions
    n_instructions = cfg.generation.n_instructions
    if n_instructions is not None:
        limit_by = resolved_task.spec.protocol.limit_by
        if limit_by is None:
            instructions_df = instructions_df.head(n_instructions)
        else:
            selected = instructions_df[limit_by].drop_duplicates().head(n_instructions)
            instructions_df = instructions_df.loc[
                instructions_df[limit_by].isin(selected)
            ]

    judge, default_temperature = prepare_pairwise_judging(
        cfg, resolved_prompt, instructions_df
    )

    baseline_plan = resolve_baseline_plan(
        task_id=cfg.task,
        task=resolved_task,
        runtime_baseline=cfg.model.baseline,
        instructions=instructions_df,
    )

    name = f"{cfg.task}-{cfg.model.name}-{baseline_plan.display_name}-{cfg.judge.model}"
    name += f"-{cfg.judge.swap_mode}"
    name = name.replace("/", "_")
    run_ts = run_started_at.strftime("%Y%m%d_%H%M%S")
    res_folder = prepare_run_directory(
        cfg, Path(cfg.run.result_folder) / f"{name}-{run_ts}"
    )

    logger.info(
        "Using task %s and evaluating %s against baseline %s.",
        cfg.task,
        cfg.model.name,
        baseline_plan.display_name,
    )

    logger.info(
        "Generating completions for task %s with model %s and baseline %s "
        "(or loading them directly if present)",
        cfg.task,
        cfg.model.name,
        baseline_plan.display_name,
    )

    comparisons = prepare_comparisons(cfg, task_data, instructions_df, baseline_plan)
    judge_chat_model = build_judge(cfg, default_temperature=default_temperature)
    annotations, outcomes, judgment_metadata = judge(judge_chat_model, comparisons)
    judge_prompt_variants = judgment_metadata["judge_prompts"]
    prefs = outcomes["pref"].rename(None)
    annotations["judge"] = cfg.judge.model
    write_annotations(annotations, res_folder / f"{name}-annotations.csv")

    battles = judged_battles(
        outcomes,
        evaluation_model=cfg.model.name,
        judge=cfg.judge.model,
        judge_temperature=cfg.judge.temperature,
        judge_max_out_tokens=cfg.judge.max_out_tokens,
    )
    metric_results = calculate_metrics(
        battles,
        metrics,
        runtime_by_metric={
            "bradley_terry": {"rng": np.random.default_rng(cfg.run.seed)},
        },
    )

    report = BattleReport(
        task=cfg.task,
        model_a=cfg.model.name,
        model_b=baseline_plan.display_name,
        judge_model=cfg.judge.model,
        metrics=metric_results,
        swap_mode=cfg.judge.swap_mode,
        result_folder=str(res_folder),
        preferences=prefs.tolist(),
        metadata={
            "baseline_assignment": "per-row"
            if not baseline_plan.is_single_model
            else "flat",
            "baseline_models": baseline_plan.unique_models,
            **resolved_prompt.metadata(),
            **judgment_metadata,
            "strip_thinking_before_judging": cfg.judge.strip_thinking_before_judging,
            "battle_thinking_token_budget": cfg.judge.battle_thinking_token_budget,
        },
    )
    results = report.to_dict()
    logger.info(
        "%s vs %s judged by %s",
        cfg.model.name,
        baseline_plan.display_name,
        cfg.judge.model,
    )
    report.render()
    report.save(res_folder / f"results-{name}.json")

    eval_instruction_index = annotations["instruction_index"].drop_duplicates().tolist()
    evaluated = comparisons.loc[eval_instruction_index]

    write_run_metadata_safely(
        task=cfg.resolved_task,
        output_dir=res_folder,
        entrypoint="judgearena.benchmarks.pairwise.runner.run_pairwise",
        run=cfg.model_dump(),
        results=results,
        input_payloads={
            "instruction_index": eval_instruction_index,
            "instructions": evaluated["instruction"].tolist(),
            "completions_A": evaluated["completion_a"].tolist(),
            "completions_B": evaluated["completion_b"].tolist(),
            "baseline_model_B": evaluated["model_b"].tolist(),
            **{
                column: evaluated[column].tolist()
                for column in ("messages", "messages_a", "messages_b")
                if column in evaluated
            },
        },
        judge_system_prompt=resolved_prompt.system_prompt,
        judge_user_prompt_template=resolved_prompt.user_prompt_template,
        judge_prompt_variants=judge_prompt_variants,
        started_at_utc=run_started_at,
    )

    return prefs
