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
    safe_filename,
    write_run_metadata_safely,
)
from judgearena.benchmarks.arena import resolve_task_languages
from judgearena.benchmarks.execution import build_judge
from judgearena.benchmarks.meta_eval.annotate import (
    aggregate_battle_preferences,
    annotate_sample,
)
from judgearena.benchmarks.meta_eval.sampling import (
    MetaEvalSamplingError,
    sample_battles_per_model,
    select_top_models,
)
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.datasets import load_battles
from judgearena.evaluate import resolve_run_judge_prompt
from judgearena.log import get_logger
from judgearena.reports import MetaEvalReport
from judgearena.tasks.schema import MetaEvalProtocol, ResolvedTaskSpec

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)

_WINNER_PREFERENCES = {
    "model_a": 0.0,
    "model_b": 1.0,
    "tie": 0.5,
    "tie (bothbad)": 0.5,
}
_REQUIRED_COLUMNS = {
    "question_id",
    "model_a",
    "model_b",
    "winner",
    "lang",
    "conversation_a",
    "conversation_b",
}


def _prepare_arena_battles(
    battles: pd.DataFrame, *, task: str, arena: str, languages: list[str]
) -> pd.DataFrame:
    """Validate arena identity fields and add the metric reference preference."""
    missing = sorted(_REQUIRED_COLUMNS - set(battles.columns))
    if missing:
        raise MetaEvalSamplingError(
            f"Task {task!r} is missing required battle columns: {missing}."
        )

    battles = battles.copy()
    identity_columns = ("question_id", "model_a", "model_b", "winner", "lang")
    for column in identity_columns:
        if battles[column].isna().any():
            raise MetaEvalSamplingError(
                f"Task {task!r} contains null values in {column}."
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

    battles["reference_pref"] = battles["winner"].map(_WINNER_PREFERENCES)
    invalid_winners = sorted(
        {
            str(value)
            for value in battles.loc[battles["reference_pref"].isna(), "winner"]
        }
    )
    if invalid_winners:
        raise MetaEvalSamplingError(
            f"Task {task!r} contains invalid human winners: {invalid_winners}."
        )

    if languages:
        battles = battles.loc[battles["lang"].isin(languages)].copy()
    if battles.empty:
        raise MetaEvalSamplingError(
            f"Task {task!r} has no battles in languages {languages}."
        )
    return battles.reset_index(drop=True)


def _metric_rng(seed: int, metric_name: str) -> np.random.Generator:
    payload = f"{seed}\0{metric_name}".encode()
    metric_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(metric_seed)


def run_meta_eval(
    cfg: RunConfig, task: ResolvedTaskSpec | None = None
) -> dict[str, object]:
    """Sample arena battles, judge them, and run configured meta metrics."""
    protocol = task.spec.protocol if task is not None else None
    if not isinstance(protocol, MetaEvalProtocol):
        raise ValueError(f"Task {cfg.task!r} does not define a meta-eval protocol.")
    if cfg.meta_eval is None:
        raise ValueError(f"Task {cfg.task!r} requires meta-eval runtime settings.")

    run_started_at = datetime.now(UTC)
    languages = resolve_task_languages(
        task, cfg.meta_eval.languages, setting="meta_eval.languages"
    )
    metrics = build_metrics(protocol.scoring.metrics)
    for request, metric in metrics:
        if request.metric == "meta_eval_elo_gap":
            maximum = max(metric.battle_counts)
            if maximum > cfg.meta_eval.battles_per_model:
                raise ValueError(
                    "The maximum meta_eval_elo_gap battle budget "
                    f"({maximum}) exceeds meta_eval.battles_per_model "
                    f"({cfg.meta_eval.battles_per_model})."
                )

    logger.info("Loading human battles from %s", protocol.arena)
    arena_battles = _prepare_arena_battles(
        load_battles(task), task=cfg.task, arena=protocol.arena, languages=languages
    )
    top_models, top_pool = select_top_models(
        arena_battles, top_models=cfg.meta_eval.top_models
    )
    sample = sample_battles_per_model(
        top_pool,
        top_models,
        battles_per_model=cfg.meta_eval.battles_per_model,
        seed=cfg.run.seed,
    )
    resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)
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
    sample.loc[
        :, ("battle_id", "question_id", "model_a", "model_b", "winner", "lang")
    ].to_parquet(result_dir / "sample.parquet", index=False)

    annotations = annotate_sample(
        sample,
        cfg,
        judge_chat_model=build_judge(cfg),
        resolved_prompt=resolved_prompt,
    )
    annotations.to_parquet(result_dir / "annotations.parquet", index=False)
    judged_battles = aggregate_battle_preferences(
        annotations, swap_mode=cfg.judge.swap_mode
    )
    metric_battles = top_pool.loc[
        :, ("battle_id", "model_a", "model_b", "reference_pref")
    ].copy()
    metric_battles["sampled"] = metric_battles["battle_id"].isin(
        judged_battles["battle_id"]
    )
    metric_battles = metric_battles.merge(
        judged_battles, on="battle_id", how="left"
    ).sort_values("battle_id", kind="stable", ignore_index=True)
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
        languages=languages,
        top_models=top_models,
        n_sampled_battles=len(sample),
        swap_mode=cfg.judge.swap_mode,
        metrics=metric_results,
    )
    results = report.to_dict()
    report.render()
    result_path = report.save(result_dir / "results.json")
    write_run_metadata_safely(
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
