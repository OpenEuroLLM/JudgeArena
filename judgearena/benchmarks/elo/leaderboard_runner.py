"""Evaluate one candidate using a leaderboard's saved benchmark inputs.

Generate candidate answers, judge them against the saved opponent answers, and
fit only the candidate rating. Save its entry and battle results, then update
the local leaderboard index without changing the reference ratings or panel.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from judgearena.artifacts import (
    prepare_run_directory,
    safe_filename,
    write_run_metadata_safely,
)
from judgearena.battles import Leaderboard, RatingEntry, write_battles
from judgearena.benchmarks.elo.artifacts import BATTLE_COLUMNS
from judgearena.benchmarks.elo.execution import (
    build_candidate_battles,
    judge_candidate_battles,
)
from judgearena.benchmarks.elo.leaderboard import (
    AnchorSet,
    collapse_swapped_rows,
    comparable_config,
    load_frozen_files,
    rebuild_leaderboard,
    write_entry,
)
from judgearena.benchmarks.elo.scoring import FrozenBradleyTerryResult
from judgearena.benchmarks.execution import (
    build_completion_cache,
    build_generation_kwargs,
)
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.config import RunConfig
from judgearena.evaluate import resolve_run_judge_prompt
from judgearena.generate import generate_instructions
from judgearena.log import get_logger
from judgearena.reports import EloReport
from judgearena.tasks.schema import ResolvedTaskSpec

logger = get_logger(__name__)


def _prepare_frozen_run(
    cfg: RunConfig, task: ResolvedTaskSpec
) -> tuple[Path, AnchorSet, pd.DataFrame]:
    """Check runtime settings before a frozen leaderboard run."""
    rating_request = next(
        (
            request
            for request in task.spec.protocol.scoring.metrics
            if request.metric == "bradley_terry"
        ),
        None,
    )
    if rating_request is None:
        raise ValueError("Frozen leaderboards require the bradley_terry metric.")
    if rating_request.breakdown_by:
        raise ValueError(
            "Frozen Bradley-Terry already includes per-language estimates; "
            "breakdown_by is not supported for this metric."
        )
    directory = Path(cfg.elo.leaderboard_dir)
    anchors, panel, frozen_config = load_frozen_files(directory)
    if anchors.task != task.task:
        raise ValueError(
            f"Frozen leaderboard task {anchors.task!r} does not match {task.task!r}."
        )
    if tuple(cfg.elo.languages or ()) != anchors.languages:
        raise ValueError("elo.languages must exactly match the frozen languages.")
    if (
        cfg.generation.n_instructions is not None
        or cfg.elo.n_instructions_per_language is not None
        or cfg.elo.elo_random_battles is not None
    ):
        raise ValueError(
            "Frozen leaderboard runs cannot resample or truncate the panel."
        )
    if cfg.elo.calibrate_temperature:
        raise ValueError("Frozen leaderboard runs cannot recalibrate temperature.")
    if comparable_config(cfg) != comparable_config(frozen_config):
        raise ValueError("runtime config does not match the frozen leaderboard config")
    leaderboard_path = rebuild_leaderboard(directory, anchors)
    leaderboard = json.loads(leaderboard_path.read_text())
    if any(row["model"] == cfg.model.name for row in leaderboard["entries"]):
        raise ValueError(f"Model {cfg.model.name!r} is already on the leaderboard.")
    return directory, anchors, panel


def run_frozen_leaderboard(cfg: RunConfig, task: ResolvedTaskSpec) -> dict:
    """Run the frozen flow after run_elo resolves the task's Elo settings."""
    protocol = task.spec.protocol
    arena = protocol.arena
    run_started_at = datetime.now(UTC)
    logger.info("Step 1: Loading frozen leaderboard panel")
    leaderboard_dir, anchors, panel = _prepare_frozen_run(cfg, task)
    panel = panel.set_index("panel_id", drop=False)
    resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)
    n = len(panel)
    sampling_metadata = {
        "sampling_mode": "frozen_panel",
        "protocol_id": anchors.protocol_id,
    }

    logger.info("Step 2: Generating completions with %s", cfg.model.name)
    completions = generate_instructions(
        instructions=panel["instruction"],
        model=cfg.model.name,
        truncate_input_chars=cfg.generation.truncate_all_input_chars,
        use_tqdm=False,
        inference_cache=build_completion_cache(cfg),
        **build_generation_kwargs(cfg, cfg.model.name, role="A"),
    ).set_index("instruction_index")["completion"]

    logger.info("Step 3: Judge evaluation with %s", cfg.judge.model)
    judged = judge_candidate_battles(cfg, panel, completions, resolved_prompt)
    df_llm_judge = build_candidate_battles(
        cfg,
        panel,
        completions,
        judged,
        parser=resolved_prompt.parser,
        effective_temperature=cfg.elo.soft_elo_temperature,
    )
    passes = 2 if cfg.judge.swap_mode == "both" else 1
    df_llm_judge["panel_id"] = panel["panel_id"].tolist() * passes
    df_llm_judge["lang"] = panel["lang"].tolist() * passes
    metric_battles = collapse_swapped_rows(df_llm_judge, cfg.judge.swap_mode)
    sampling_metadata["attempted_battles"] = n
    sampling_metadata["skipped_battles"] = n - len(metric_battles)
    model_name = cfg.model.name
    metric_battles["evaluation_model"] = model_name
    metrics = build_metrics(
        protocol.scoring.metrics,
        parameter_overrides_by_metric={
            "bradley_terry": {
                "n_bootstraps": cfg.elo.n_bootstraps,
                "soft": cfg.elo.soft_elo,
            },
        },
    )
    metric_results = calculate_metrics(
        metric_battles,
        metrics,
        runtime_by_metric={"bradley_terry": {"anchors": anchors}},
    )
    rating_result = FrozenBradleyTerryResult.model_validate(
        metric_results["bradley_terry"]
    )
    leaderboard_entry = rating_result.entry
    report = EloReport(
        arena=arena,
        judge_model=cfg.judge.model,
        metrics=metric_results,
        num_battles=len(metric_battles),
        model_name=model_name,
        sampling_metadata=sampling_metadata,
    )
    results = report.to_dict()
    report.render()
    model_digest = hashlib.sha256(model_name.encode()).hexdigest()[:16]
    result_directory = (
        Path(cfg.run.result_folder)
        / f"elo-{safe_filename(arena)}-{safe_filename(model_name)}-"
        f"{safe_filename(cfg.judge.model)}"
        / anchors.protocol_id
        / f"model-{model_digest}"
    )
    res_dir = prepare_run_directory(cfg, result_directory)
    result_path = report.save(res_dir / f"results-{safe_filename(model_name)}.json")
    write_battles(res_dir / "battles.parquet", df_llm_judge[list(BATTLE_COLUMNS)])
    summary = leaderboard_entry.overall
    if summary.ci_low is not None:
        Leaderboard(
            arena=arena,
            model=model_name,
            judge_model=cfg.judge.model,
            n_bootstraps=rating_result.n_bootstraps,
            seed=anchors.bootstrap_seed,
            ratings=[
                RatingEntry(
                    model=model_name,
                    rating=summary.rating,
                    ci_low=summary.ci_low,
                    ci_high=summary.ci_high,
                    n_battles=summary.n_battles,
                    source="evaluated",
                )
            ],
        ).write(res_dir / "elo_ratings.json")
    write_run_metadata_safely(
        output_dir=res_dir,
        entrypoint="judgearena.benchmarks.elo.runner.run_elo",
        run=cfg.model_dump(),
        results=results,
        input_payloads={"question_id": panel["question_id"].tolist()},
        judge_system_prompt=resolved_prompt.system_prompt,
        judge_user_prompt_template=resolved_prompt.user_prompt_template,
        started_at_utc=run_started_at,
    )
    (res_dir / "entry.json").write_text(
        leaderboard_entry.model_dump_json(indent=2) + "\n"
    )
    entry_path = write_entry(leaderboard_dir, leaderboard_entry)
    leaderboard_path = rebuild_leaderboard(leaderboard_dir, anchors)
    return {
        **results,
        "result_path": str(result_path),
        "leaderboard_entry_path": str(entry_path),
        "leaderboard_path": str(leaderboard_path),
    }
