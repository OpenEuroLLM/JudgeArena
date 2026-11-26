"""Shared answer generation, judging, metrics, and artifacts for model evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from judgearena.artifacts import (
    prepare_run_directory,
    run_logging,
    write_annotations,
    write_run_metadata_safely,
)
from judgearena.artifacts.evaluation import finish_evaluation
from judgearena.battles import judged_battles
from judgearena.benchmarks.execution import build_judge
from judgearena.benchmarks.pairwise.baselines import resolve_baseline_plan
from judgearena.benchmarks.pairwise.comparisons import (
    preload_model_outputs,
    prepare_answers,
    prepare_comparisons,
    select_instructions,
)
from judgearena.benchmarks.pairwise.graph import (
    prepare_candidate_comparisons,
    prepare_recorded_opponents,
    recorded_references,
    validate_graph_task,
)
from judgearena.benchmarks.pairwise.judging import prepare_pairwise_judging
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.datasets import load_task_data, resolve_dataset_filters
from judgearena.datasets.data import PreferenceData, with_instruction_metadata
from judgearena.evaluate import (
    prepare_single_answer_judging,
    resolve_run_judge_prompt,
)
from judgearena.judging import answer_samples
from judgearena.log import get_logger
from judgearena.prompts.registry import prompt_metadata
from judgearena.reports import AnnotationReport, BattleReport, ComparisonReport
from judgearena.tasks.schema import RecordedOpponents, ResolvedTaskSpec

if TYPE_CHECKING:
    from judgearena.benchmarks.pairwise.baselines import BaselinePlan
    from judgearena.benchmarks.scoring import ConfiguredMetrics
    from judgearena.config import RunConfig
    from judgearena.datasets.data import CompletionData

logger = get_logger(__name__)


def _answer_columns(data, generation_mode):
    columns = {"model", "completion"}
    if data.model_outputs is not None:
        columns.update(set(data.model_outputs) - {"instruction_index"})
    if generation_mode in (
        "multi_turn_chat",
        "provided_messages",
    ):
        columns.add("messages")
    return columns


def prepare_independent_judge(cfg, data, instructions, prompt=None):
    """Validate single-answer prompts against the fields preparation can supply."""
    if prompt is None:
        prompt = resolve_run_judge_prompt(
            cfg.resolved_task,
            cfg.judge,
            multi_turn=cfg.resolved_task.spec.protocol.generation.mode
            == "multi_turn_chat",
        )
    columns = set(instructions) | _answer_columns(
        data, cfg.resolved_task.spec.protocol.generation.mode
    )
    if "messages" in instructions:
        columns.add("input_messages")
    return prepare_single_answer_judging(
        cfg, prompt, instructions, available_columns=columns
    )


@run_logging()
def run_evaluation(
    cfg: RunConfig,
    resolved_task: ResolvedTaskSpec | None = None,
    *,
    data: CompletionData | None = None,
    output_dir: str | Path | None = None,
    baseline_plan: BaselinePlan | None = None,
    opponent_panel: pd.DataFrame | None = None,
    metrics: ConfiguredMetrics | None = None,
    sampling_metadata: dict[str, object] | None = None,
) -> Path | pd.Series | pd.DataFrame:
    """Load or generate answers, judge them, and save annotations and metrics.

    Each opponent_panel row identifies a prompt by instruction_index and supplies
    one saved opponent answer in model_b and completion_b.
    """

    run_started_at = datetime.now(UTC)

    cfg = cfg.for_task(resolved_task)
    resolved_task = cfg.resolved_task
    independent = resolved_task.spec.protocol.runner == "single_answer"
    resolved_prompt = resolve_run_judge_prompt(
        resolved_task,
        cfg.judge,
        multi_turn=resolved_task.spec.protocol.generation.mode == "multi_turn_chat",
    )
    if metrics is None:
        metrics = build_metrics(resolved_task.spec.protocol.scoring.metrics)
    task_data = data if data is not None else load_task_data(resolved_task)
    comparison_rng = np.random.default_rng(cfg.run.seed)
    if opponent_panel is not None and cfg.model.baseline is not None:
        raise ValueError("Choose either model.baseline or a supplied opponent panel.")
    recorded = isinstance(resolved_task.spec.protocol.baseline, RecordedOpponents)
    should_sample_opponents = (
        recorded
        and opponent_panel is None
        and baseline_plan is None
        and cfg.model.baseline is None
    )
    reference_battles = (
        recorded_references(task_data)
        if isinstance(task_data, PreferenceData)
        and (recorded or opponent_panel is not None)
        else None
    )
    if should_sample_opponents:
        opponent_panel, sampling_metadata = prepare_recorded_opponents(
            cfg, task_data, rng=comparison_rng
        )
    if opponent_panel is None:
        task_data = task_data.select(
            resolve_dataset_filters(resolved_task, cfg.run.filters)
        )
        instructions_df = select_instructions(cfg, task_data.instructions)
        if not independent and baseline_plan is None:
            baseline_plan = resolve_baseline_plan(
                task_id=cfg.task,
                task=resolved_task,
                runtime_baseline=cfg.model.baseline,
                instructions=instructions_df,
            )
    else:
        validate_graph_task(resolved_task.spec.protocol)
        if "model_a" in opponent_panel or not {
            "model_b",
            "completion_b",
            "instruction_index",
        } <= set(opponent_panel):
            raise ValueError(
                "Each panel row must supply instruction_index, model_b and completion_b for one opponent."
            )
        instruction_ids = pd.Index(opponent_panel["instruction_index"])
        instructions_df = task_data.instructions.loc[instruction_ids.unique()]
        if not should_sample_opponents:
            opponent_panel = with_instruction_metadata(opponent_panel, instructions_df)
    task_data = preload_model_outputs(
        task_data,
        [cfg.model.name] + (baseline_plan.unique_models if baseline_plan else []),
    )

    name = f"{cfg.task}-{cfg.model.name}-{cfg.judge.model}"
    if independent:
        judge = prepare_independent_judge(
            cfg, task_data, instructions_df, resolved_prompt
        )
    else:
        context = opponent_panel if opponent_panel is not None else instructions_df
        output_columns = _answer_columns(
            task_data, resolved_task.spec.protocol.generation.mode
        )
        available_columns = set(context) | {
            f"{column}_{slot}" for column in output_columns for slot in ("a", "b")
        }
        if opponent_panel is not None:
            available_columns.update(
                ("instruction_id", "question_id", "cache_orientation", "source")
            )
        if resolved_task.spec.protocol.judge.turns is not None:
            available_columns.update(
                ("turn", "question_id", "source_instruction_index")
            )
        judge, default_temperature = prepare_pairwise_judging(
            cfg,
            resolved_prompt,
            context,
            available_columns=available_columns,
            reference_battles=reference_battles,
            calibration_rng=comparison_rng if should_sample_opponents else None,
            # Sampled battles already randomize candidate orientation.
            swap_mode="fixed"
            if should_sample_opponents and cfg.judge.swap_mode != "both"
            else None,
        )
        if baseline_plan is not None:
            name = f"{cfg.task}-{cfg.model.name}-{baseline_plan.display_name}-{cfg.judge.model}-{cfg.judge.swap_mode}"

    name = name.replace("/", "_")
    run_ts = run_started_at.strftime("%Y%m%d_%H%M%S")
    res_folder = prepare_run_directory(
        cfg,
        output_dir
        if output_dir is not None
        else Path(cfg.run.result_folder) / f"{name}-{run_ts}",
    )

    logger.info(
        "Evaluating %s on %s with judge %s.", cfg.model.name, cfg.task, cfg.judge.model
    )
    answers = prepare_answers(cfg, task_data, instructions_df, cfg.model.name, role="A")
    if independent:
        samples = answer_samples(instructions_df, answers)
        annotations, prompts = judge(build_judge(cfg), samples)
        judgment_metadata = {"judge_prompts": prompt_metadata(prompts)}
    elif opponent_panel is not None:
        samples, source_ids = prepare_candidate_comparisons(
            opponent_panel,
            answers,
            rng=comparison_rng if should_sample_opponents else None,
            turns=resolved_task.spec.protocol.judge.turns,
        )
    else:
        samples, source_ids = prepare_comparisons(
            cfg, task_data, instructions_df, baseline_plan, answers=answers
        )
    if not independent:
        judge_chat_model = build_judge(cfg, default_temperature=default_temperature)
        annotations, outcomes, judgment_metadata = judge(
            judge_chat_model, samples, source_ids=source_ids
        )
    judge_prompt_variants = judgment_metadata["judge_prompts"]
    annotations = annotations.assign(judge=cfg.judge.model)
    write_annotations(annotations, res_folder / f"{name}-annotations.csv")

    metric_data = (
        annotations
        if independent
        else judged_battles(
            outcomes,
            evaluation_model=cfg.model.name,
            judge=cfg.judge.model,
            judge_temperature=cfg.judge.temperature,
            judge_max_out_tokens=cfg.judge.max_out_tokens,
        )
    )
    reference_scores = None
    if reference_battles is not None:
        # Calibration needs transcripts; reference scoring needs labels and metadata.
        reference_scores = reference_battles.drop(
            columns=[
                "instruction",
                "user_turns",
                "messages",
                "completion_a",
                "completion_b",
                "messages_a",
                "messages_b",
                "conversation_a",
                "conversation_b",
            ],
            errors="ignore",
        )
    metric_results = calculate_metrics(
        metric_data,
        metrics,
        runtime={
            "seed": cfg.run.seed,
            "reference_battles": reference_scores,
            "references": (
                baseline_plan.unique_models
                if baseline_plan
                else sorted(
                    set(reference_battles.model_a) | set(reference_battles.model_b)
                )
                if reference_battles is not None
                else sorted(metric_data.baseline.unique())
                if opponent_panel is not None
                else []
            ),
        },
    )

    metadata = {
        **resolved_prompt.metadata(),
        **judgment_metadata,
        "strip_thinking_before_judging": cfg.judge.strip_thinking_before_judging,
        "battle_thinking_token_budget": cfg.judge.battle_thinking_token_budget,
    }
    if independent:
        report = AnnotationReport(
            task=cfg.task,
            model=cfg.model.name,
            judge_model=cfg.judge.model,
            metrics=metric_results,
            num_samples=len(annotations),
            metadata=metadata,
            result_folder=str(res_folder),
        )
        result = annotations
    elif opponent_panel is not None:
        result = outcomes["pref"].rename(None)
        report = ComparisonReport(
            arena=resolved_task.task,
            judge_model=cfg.judge.model,
            metrics=metric_results,
            num_battles=len(opponent_panel),
            model_name=cfg.model.name,
            sampling_metadata={**(sampling_metadata or {}), **judgment_metadata},
        )
    else:
        result = outcomes["pref"].rename(None)
        report = BattleReport(
            task=cfg.task,
            model_a=cfg.model.name,
            model_b=baseline_plan.display_name,
            judge_model=cfg.judge.model,
            metrics=metric_results,
            swap_mode=cfg.judge.swap_mode,
            result_folder=str(res_folder),
            preferences=result.tolist(),
            metadata={
                "baseline_models": baseline_plan.unique_models,
                **metadata,
            },
        )
    results = report.to_dict()
    report.render(metrics)
    report_path = report.save(
        res_folder
        / ("results.json" if output_dir is not None else f"results-{name}.json")
    )

    eval_instruction_index = annotations["instruction_index"].drop_duplicates().tolist()
    evaluated = samples.loc[eval_instruction_index]

    write_run_metadata_safely(
        task=cfg.resolved_task,
        output_dir=res_folder,
        entrypoint="judgearena.benchmarks.evaluation.run_evaluation",
        run=cfg.model_dump(),
        results=results,
        input_payloads={
            "instruction_index": eval_instruction_index,
            "instructions": evaluated["instruction"].tolist(),
            **(
                {"completions": evaluated["completion"].tolist()}
                if independent
                else {
                    "completions_A": evaluated["completion_a"].tolist(),
                    "completions_B": evaluated["completion_b"].tolist(),
                    "baseline_model_B": evaluated["model_b"].tolist(),
                }
            ),
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

    write_annotations(
        answers.rename_axis("instruction_index").reset_index(),
        res_folder / "answers.parquet",
    )
    write_annotations(annotations, res_folder / "annotations.parquet")
    metric_path = res_folder / "annotations.parquet"
    if not independent:
        metric_path = res_folder / "outcomes.parquet"
        write_annotations(metric_data, metric_path)
    artifacts = {
        "report": report_path,
        "metric_data": metric_path,
        "answers": res_folder / "answers.parquet",
        "annotations": res_folder / "annotations.parquet",
    }
    if reference_scores is not None:
        artifacts["reference_battles"] = res_folder / "reference-battles.parquet"
        write_annotations(reference_scores, artifacts["reference_battles"])
    manifest = finish_evaluation(
        res_folder,
        report,
        model=cfg.model.name,
        artifacts=artifacts,
    )
    return manifest if output_dir is not None else result
