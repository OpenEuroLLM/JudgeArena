import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from judgearena.artifacts import (
    prepare_run_directory,
    run_logging,
    safe_filename,
)
from judgearena.artifacts.evaluation import finish_evaluation
from judgearena.battles import (
    Leaderboard,
    RatingEntry,
    judged_battles,
    write_battles,
)
from judgearena.benchmarks.elo.scoring import elo_metrics
from judgearena.benchmarks.evaluation import run_evaluation
from judgearena.benchmarks.pairwise.graph import (
    prepare_graph,
)
from judgearena.datasets import load_task_data
from judgearena.datasets.data import PreferenceData
from judgearena.evaluate import (
    resolve_run_judge_prompt,
)
from judgearena.log import get_logger
from judgearena.reports import ComparisonReport, EloReport  # noqa: F401
from judgearena.tasks.schema import EloProtocol, ResolvedTaskSpec

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


@run_logging()
def run_elo(
    cfg: "RunConfig",
    task: ResolvedTaskSpec | None = None,
    *,
    data=None,
    output_dir=None,
    opponent_panel=None,
    **evaluation_options,
) -> dict | Path:
    """Rate one model against the reference battles defined by an ELO task."""
    cfg = cfg.for_task(task)
    if opponent_panel is not None:
        return run_evaluation(
            cfg,
            data=data,
            output_dir=output_dir,
            opponent_panel=opponent_panel,
            **evaluation_options,
        )
    if evaluation_options:
        raise TypeError("Supplied evaluation options require an opponent_panel")
    task = cfg.resolved_task
    protocol = task.spec.protocol
    if not isinstance(protocol, EloProtocol):
        raise ValueError(f"Task {cfg.task!r} does not define an ELO protocol.")
    if cfg.elo is None:
        raise ValueError(f"Task {cfg.task!r} requires ELO runtime settings.")
    cfg.elo = cfg.elo.resolve(protocol.scoring)
    resolved_prompt = resolve_run_judge_prompt(task, cfg.judge)
    arena = protocol.arena
    run_started_at = datetime.now(UTC)
    rng = np.random.default_rng(cfg.run.seed)
    folder_name = (
        f"elo-{safe_filename(arena)}-{safe_filename(cfg.model.name)}-"
        f"{safe_filename(cfg.judge.model)}-{run_started_at:%Y%m%d_%H%M%S_%f}"
    )
    res_dir = prepare_run_directory(
        cfg, output_dir or Path(cfg.run.result_folder) / folder_name
    )

    logger.info("Loading battles from %s", arena)
    task_data = data if data is not None else load_task_data(task)
    if not isinstance(task_data, PreferenceData):
        raise ValueError(
            "Anchored Elo requires PreferenceData with reference preferences."
        )
    df_battles, references, sampling_metadata = prepare_graph(cfg, task_data)
    if references.empty:
        raise ValueError(
            "Legacy runner: elo requires observed reference preferences. "
            "Use runner: pairwise with graph settings for unlabeled comparisons."
        )
    logger.info("Loaded %d battles.", len(df_battles))

    manifest_path = run_evaluation(
        cfg,
        data=task_data,
        prepared_directory=res_dir,
        report_name=f"results-{safe_filename(cfg.model.name)}.json",
        opponent_panel=df_battles,
        metrics=elo_metrics(cfg),
        reference_battles=references,
        sampling_metadata=sampling_metadata,
        comparison_rng=rng,
        prompt=resolved_prompt,
    )
    manifest = json.loads(manifest_path.read_text())
    res_dir = manifest_path.parent
    result_path = res_dir / manifest["artifacts"]["report"]
    results = json.loads(result_path.read_text())
    report = ComparisonReport(**results)
    metric_results = report.metrics
    model_name = cfg.model.name
    df_llm_judge = pd.read_parquet(res_dir / "outcomes.parquet")
    df_llm_judge = judged_battles(
        df_llm_judge,
        evaluation_model=model_name,
        judge=cfg.judge.model,
        judge_temperature=cfg.judge.temperature,
        judge_max_out_tokens=cfg.judge.max_out_tokens,
    )

    battle_cols = [
        "model_a",
        "model_b",
        "pref",
        "pref_hard",
        "source",
        "judge_model",
        "question_id",
    ]
    write_battles(
        res_dir / "battles.parquet",
        df_llm_judge[[c for c in battle_cols if c in df_llm_judge.columns]],
    )
    rating_result = metric_results.get("bradley_terry")
    if rating_result is not None and rating_result.get("rating_entries"):
        if rating_result.get("bootstrap_ratings"):
            pd.DataFrame(rating_result["bootstrap_ratings"]).to_csv(
                res_dir / "bootstrap_ratings.csv", index=False
            )
        entries = [RatingEntry(**entry) for entry in rating_result["rating_entries"]]
        Leaderboard(
            arena=arena,
            model=model_name,
            judge_model=cfg.judge.model,
            n_bootstraps=rating_result["n_bootstraps"],
            seed=cfg.run.seed,
            ratings=entries,
        ).write(res_dir / "elo_ratings.json")

    # Retain the legacy rating exports alongside the common evaluation artifacts.
    manifest_path = finish_evaluation(
        res_dir,
        report,
        model=model_name,
        artifacts={
            key: res_dir / value for key, value in manifest["artifacts"].items()
        },
    )
    return (
        manifest_path
        if output_dir is not None
        else {
            **results,
            "result_path": str(result_path),
        }
    )
