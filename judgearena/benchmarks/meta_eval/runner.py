"""Score a judge against reference-labeled comparisons."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from judgearena.artifacts import (
    prepare_run_directory,
    run_logging,
    safe_filename,
    write_annotations,
    write_run_metadata_safely,
)
from judgearena.benchmarks.execution import build_judge
from judgearena.benchmarks.meta_eval.annotate import (
    prepare_reference_comparisons,
    reference_scoring_battles,
)
from judgearena.benchmarks.meta_eval.sampling import (
    MetaEvalSamplingError,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.benchmarks.pairwise.comparisons import select_reference_battles
from judgearena.benchmarks.pairwise.judging import prepare_pairwise_judging
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.datasets import load_task_data, resolve_dataset_filters
from judgearena.datasets.data import PreferenceData
from judgearena.evaluate import resolve_run_judge_prompt
from judgearena.log import get_logger
from judgearena.reports import MetaEvalReport
from judgearena.tasks.schema import MetaEvalProtocol, ResolvedTaskSpec

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


def _prepare_arena_battles(data: PreferenceData, *, task: str) -> pd.DataFrame:
    """Select reference comparisons and retain physical battle identities."""
    if not isinstance(data, PreferenceData):
        raise MetaEvalSamplingError(
            "Meta-evaluation requires PreferenceData with reference preferences."
        )
    battles = select_reference_battles(data.battles)
    battles = battles.loc[battles["model_a"] != battles["model_b"]]
    if battles.empty:
        raise MetaEvalSamplingError(
            f"Task {task!r} has no selected battles between different models."
        )
    return battles.reset_index(drop=True)


@run_logging()
def run_meta_eval(
    cfg: RunConfig, task: ResolvedTaskSpec | None = None, *, data=None
) -> dict[str, object]:
    """Sample arena battles, judge them, and run configured meta metrics."""
    cfg = cfg.for_task(task)
    task = cfg.resolved_task
    protocol = task.spec.protocol
    if not isinstance(protocol, MetaEvalProtocol):
        raise ValueError(f"Task {cfg.task!r} does not define a meta-eval protocol.")
    if cfg.meta_eval is None:
        raise ValueError(f"Task {cfg.task!r} requires meta-eval runtime settings.")

    if protocol.judge.minimum_score_by:
        raise ValueError(
            "Meta-evaluation requires outcomes for the original reference comparisons; "
            "minimum_score_by discards their reference labels."
        )

    run_started_at = datetime.now(UTC)
    filters = resolve_dataset_filters(
        task,
        cfg.run.filters,
        {"lang": cfg.meta_eval.languages} if cfg.meta_eval.languages else {},
    )
    metrics = build_metrics(protocol.scoring.metrics)

    logger.info("Loading reference battles from %s", protocol.arena)
    data = (data if data is not None else load_task_data(task)).select(filters)
    arena_battles = _prepare_arena_battles(data, task=cfg.task)
    top_models, top_pool = select_top_models(
        arena_battles, top_models=cfg.meta_eval.top_models
    )
    sample = sample_battles_per_model(
        top_pool,
        top_models,
        battles_per_model=cfg.meta_eval.battles_per_model,
        seed=cfg.run.seed,
    )
    resolved_prompt = resolve_run_judge_prompt(cfg.resolved_task, cfg.judge)
    logger.info(
        "Sampled %d battles among the top %d models.", len(sample), len(top_models)
    )
    timestamp = run_started_at.strftime("%Y%m%d_%H%M%S")
    result_name = (
        f"{safe_filename(cfg.task)}-{safe_filename(cfg.judge.model)}-"
        f"{cfg.judge.swap_mode}-{timestamp}"
    )
    result_dir = prepare_run_directory(cfg, Path(cfg.run.result_folder) / result_name)
    write_annotations(sample, result_dir / "sample.parquet")

    comparisons = prepare_reference_comparisons(sample, turns=protocol.judge.turns)
    judge_comparisons, default_temperature = prepare_pairwise_judging(
        cfg,
        resolved_prompt,
        comparisons,
        available_columns=comparisons.columns,
    )
    judge = build_judge(cfg, default_temperature=default_temperature)
    annotations, outcomes, judgment_metadata = judge_comparisons(judge, comparisons)
    write_annotations(annotations, result_dir / "annotations.parquet")
    metric_battles = reference_scoring_battles(top_pool, outcomes)
    write_annotations(metric_battles, result_dir / "battles.parquet")
    metric_results = calculate_metrics(
        metric_battles,
        metrics,
        runtime={"rng": cfg.run.seed, "references": top_models},
    )
    report = MetaEvalReport(
        task=cfg.task,
        arena=protocol.arena,
        judge_model=cfg.judge.model,
        prompt_preset=resolved_prompt.preset_name,
        languages=filters.get("lang", []),
        top_models=top_models,
        n_sampled_battles=len(sample),
        swap_mode=cfg.judge.swap_mode,
        metrics=metric_results,
    )
    results = report.to_dict()
    report.render()
    result_path = report.save(result_dir / "results.json")
    write_run_metadata_safely(
        task=cfg.resolved_task,
        output_dir=result_dir,
        entrypoint="judgearena.benchmarks.meta_eval.runner.run_meta_eval",
        run=cfg.model_dump(),
        results=results,
        input_payloads={"battle_id": sample["battle_id"].astype(str).tolist()},
        judge_system_prompt=resolved_prompt.system_prompt,
        judge_user_prompt_template=resolved_prompt.user_prompt_template,
        judge_prompt_variants=judgment_metadata["judge_prompts"],
        started_at_utc=run_started_at,
    )
    logger.info("Meta-eval results written to %s", result_dir)
    return {**results, "result_path": str(result_path)}
