"""Score a judge against human-labeled arena battles."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from judgearena.artifacts import (
    prepare_run_directory,
    run_logging,
    safe_filename,
    write_annotations,
    write_run_metadata_safely,
)
from judgearena.benchmarks.execution import build_judge, judge_prepared_comparisons
from judgearena.benchmarks.meta_eval.annotate import (
    prepare_reference_comparisons,
    reference_scoring_battles,
)
from judgearena.benchmarks.meta_eval.sampling import (
    MetaEvalSamplingError,
    sample_battles_per_model,
    select_top_models,
)
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


def _prepare_arena_battles(
    data: PreferenceData, *, task: str, arena: str
) -> pd.DataFrame:
    """Validate human comparison labels and retain physical battle identities."""
    if not isinstance(data, PreferenceData):
        raise MetaEvalSamplingError(
            "Meta-evaluation requires PreferenceData with human labels."
        )
    battles = data.battles.copy()
    if "question_id" not in battles:
        battles["question_id"] = battles["instruction_index"]
    if "lang" not in battles:
        battles["lang"] = ""
    if not battles["source"].eq("human").all():
        raise MetaEvalSamplingError(f"Task {task!r} requires human labels.")
    if not battles["pref"].isin([0.0, 0.5, 1.0]).all():
        raise MetaEvalSamplingError(
            f"Task {task!r} requires observed human preferences of 0, 0.5, or 1."
        )
    if battles["question_id"].isna().any():
        raise MetaEvalSamplingError(
            f"Task {task!r} contains null values in question_id."
        )
    for column in ("model_a", "model_b"):
        valid = battles[column].map(
            lambda value: isinstance(value, str) and bool(value)
        )
        if not valid.all():
            raise MetaEvalSamplingError(
                f"Task {task!r} contains invalid model identifiers in {column}."
            )

    battles["battle_id"] = arena + ":" + battles["question_id"].astype(str)
    if battles["battle_id"].duplicated().any():
        raise MetaEvalSamplingError(
            f"Task {task!r} contains duplicate physical battle IDs."
        )

    battles["reference_pref"] = battles["pref"]

    battles = battles.loc[battles["model_a"] != battles["model_b"]]
    if battles.empty:
        raise MetaEvalSamplingError(
            f"Task {task!r} has no selected battles between different models."
        )
    return battles.reset_index(drop=True)


def _metric_rng(seed: int, metric_name: str) -> np.random.Generator:
    payload = f"{seed}\0{metric_name}".encode()
    metric_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(metric_seed)


@run_logging()
def run_meta_eval(
    cfg: RunConfig, task: ResolvedTaskSpec | None = None
) -> dict[str, object]:
    """Sample arena battles, judge them, and run configured meta metrics."""
    cfg = cfg.for_task(task)
    task = cfg.resolved_task
    protocol = task.spec.protocol
    if not isinstance(protocol, MetaEvalProtocol):
        raise ValueError(f"Task {cfg.task!r} does not define a meta-eval protocol.")
    if cfg.meta_eval is None:
        raise ValueError(f"Task {cfg.task!r} requires meta-eval runtime settings.")

    run_started_at = datetime.now(UTC)
    filters = resolve_dataset_filters(
        task,
        cfg.run.filters,
        {"lang": cfg.meta_eval.languages} if cfg.meta_eval.languages else {},
    )
    metrics = build_metrics(protocol.scoring.metrics)

    logger.info("Loading human battles from %s", protocol.arena)
    data = load_task_data(task).select(filters)
    arena_battles = _prepare_arena_battles(data, task=cfg.task, arena=protocol.arena)
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
    if resolved_prompt.delegated:
        raise ValueError(
            "Meta-evaluation cannot use delegated prompt preset "
            f"{resolved_prompt.preset_name!r}."
        )
    if resolved_prompt.parser is None:
        raise ValueError(
            f"Prompt preset {resolved_prompt.preset_name!r} has no judge parser."
        )
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

    judge = build_judge(cfg)
    comparisons = prepare_reference_comparisons(sample)
    annotations = judge_prepared_comparisons(
        cfg,
        judge,
        comparisons,
        resolved_prompt,
        swap_mode=cfg.judge.swap_mode,
        use_tqdm=cfg.run.use_tqdm,
    )
    write_annotations(annotations, result_dir / "annotations.parquet")
    metric_battles = reference_scoring_battles(
        top_pool, annotations, swap_mode=cfg.judge.swap_mode
    )
    metric_battles.to_parquet(result_dir / "battles.parquet", index=False)
    runtime_by_metric = {
        request.metric: {"rng": _metric_rng(cfg.run.seed, request.metric)}
        for request, _metric in metrics
    }
    metric_results = calculate_metrics(
        metric_battles, metrics, runtime_by_metric=runtime_by_metric
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
        started_at_utc=run_started_at,
    )
    logger.info("Meta-eval results written to %s", result_dir)
    return {**results, "result_path": str(result_path)}
