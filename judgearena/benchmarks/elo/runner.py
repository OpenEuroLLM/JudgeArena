from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from judgearena.arenas_utils import extract_turn_text
from judgearena.artifacts import (
    prepare_run_directory,
    safe_filename,
    write_run_metadata_safely,
)
from judgearena.battles import Leaderboard, RatingEntry, write_battles
from judgearena.benchmarks.arena import resolve_task_languages
from judgearena.benchmarks.elo.calibration import calibrate_pairscore_temperature
from judgearena.benchmarks.elo.execution import (
    build_candidate_battles,
    judge_candidate_battles,
)
from judgearena.benchmarks.elo.rating import (
    arena_anchor_battles,
    select_seeded_random_arena_battles,
)
from judgearena.benchmarks.execution import (
    build_completion_cache,
    build_generation_kwargs,
    build_judgement_cache,
)
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.datasets import load_battles
from judgearena.evaluate import resolve_run_judge_prompt
from judgearena.generate import generate_instructions
from judgearena.log import get_logger
from judgearena.models import build_default_judge_model_kwargs
from judgearena.reports import EloReport
from judgearena.tasks.schema import EloProtocol, ResolvedTaskSpec

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


def run_elo(cfg: "RunConfig", task: ResolvedTaskSpec | None = None) -> dict:
    """Rate one model against the human battles defined by an ELO task."""
    protocol = task.spec.protocol if task is not None else None
    if not isinstance(protocol, EloProtocol):
        raise ValueError(f"Task {cfg.task!r} does not define an ELO protocol.")
    if cfg.elo is None:
        raise ValueError(f"Task {cfg.task!r} requires ELO runtime settings.")
    cfg.elo = cfg.elo.resolve(protocol.scoring)
    if cfg.elo.leaderboard_dir is not None:
        from judgearena.benchmarks.elo.leaderboard_runner import run_frozen_leaderboard

        return run_frozen_leaderboard(cfg, task)
    arena = protocol.arena
    run_started_at = datetime.now(UTC)
    rng = np.random.default_rng(cfg.run.seed)

    # Step 1: Load arena battles
    logger.info("Step 1: Loading battles from %s", arena)
    df_arena_all = load_battles(task)

    # A task variant preselects languages; elo.languages may narrow it further.
    selected_languages = resolve_task_languages(
        task, cfg.elo.languages, setting="elo.languages"
    )

    df_battles = df_arena_all
    if selected_languages:
        df_battles = df_battles[df_battles["lang"].isin(selected_languages)]

    random_sampling = cfg.elo.elo_random_battles is not None
    sampling_metadata: dict[str, object] = {"sampling_mode": "head"}
    if random_sampling:
        if (
            cfg.generation.n_instructions is not None
            or cfg.elo.n_instructions_per_language is not None
        ):
            raise ValueError(
                "n_instructions and n_instructions_per_language cannot be combined "
                "with elo_random_battles."
            )
        df_battles, sampling_metadata = select_seeded_random_arena_battles(
            df_battles,
            n_battles=cfg.elo.elo_random_battles,
            seed=cfg.run.seed,
        )
    else:
        # Keep at most n_instructions_per_language per language
        if cfg.elo.n_instructions_per_language is not None:
            df_battles = (
                df_battles.groupby("lang")
                .head(cfg.elo.n_instructions_per_language)
                .reset_index(drop=True)
            )

        # Keep at most n_instructions total (subset used for LLM-judge evaluation)
        if cfg.generation.n_instructions is not None:
            df_battles = df_battles.head(cfg.generation.n_instructions)

    df_battles = df_battles.reset_index(drop=True)
    n = len(df_battles)
    logger.info("Loaded %d battles.", n)

    # Extract user instructions (first turn of conversation_a)
    instructions = pd.Series(
        [
            extract_turn_text(row["conversation_a"][0])
            for _, row in df_battles.iterrows()
        ],
        index=(
            arena + ":" + df_battles["question_id"].astype(str)
            if "question_id" in df_battles
            else df_battles.index.astype(str)
        ),
        name="instruction",
    )
    logger.debug("First instruction:\n%s", instructions.iloc[0][:300])

    # Step 2: Generate completions for the model under evaluation
    logger.info("Step 2: Generating completions with %s", cfg.model.name)

    # Mirror the benchmark generation path so Elo battles honor the
    # thinking-token sub-budget for thinking models (the Elo entrypoint
    # previously called evaluated_generation_kwargs() directly and silently
    # dropped battle_thinking_token_budget).
    extra_kwargs = build_generation_kwargs(cfg, cfg.model.name, role="A")
    use_tqdm = False
    completions_df = generate_instructions(
        instructions=instructions,
        model=cfg.model.name,
        truncate_input_chars=cfg.generation.truncate_all_input_chars,
        use_tqdm=use_tqdm,
        inference_cache=build_completion_cache(cfg),
        **extra_kwargs,
    ).set_index("instruction_index")
    completions = completions_df.loc[:, "completion"]

    logger.debug("First completion:\n%s", completions.iloc[0])

    # Step 3: Judge evaluation against randomly picked arena opponents
    logger.info("Step 3: Judge evaluation with %s", cfg.judge.model)

    # For each battle, randomly pick opponent: model_a or model_b from the arena
    use_model_a_as_opponent = rng.choice([True, False], size=n)
    # Randomly decide if our model is in position A or B for the judge
    our_model_is_position_a = rng.choice([True, False], size=n)

    opponent_completions = [
        (
            extract_turn_text(row["conversation_a"][1])
            if use_model_a_as_opponent[i]
            else extract_turn_text(row["conversation_b"][1])
        )
        for i, (_, row) in enumerate(df_battles.iterrows())
    ]
    opponent_models = [
        row["model_a"] if use_model_a_as_opponent[i] else row["model_b"]
        for i, (_, row) in enumerate(df_battles.iterrows())
    ]

    # Keep request order and instruction IDs, including repeated question IDs.
    panel = pd.DataFrame(
        {
            "instruction": instructions.tolist(),
            "question_id": (
                df_battles["question_id"].tolist()
                if "question_id" in df_battles
                else [None] * n
            ),
            "opponent_model": opponent_models,
            "opponent_completion": opponent_completions,
            "candidate_position": np.where(our_model_is_position_a, "A", "B"),
        },
        index=instructions.index,
    )
    resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)
    judged = judge_candidate_battles(cfg, panel, completions, resolved_prompt)
    logger.debug("First judge output:\n%s", judged[0][0].judge_completion[:500])

    judge_extra_kwargs = build_default_judge_model_kwargs(
        cfg.judge.model,
        cfg.model.engine_kwargs,
        judge_engine_kwargs_override=cfg.judge.model_kwargs(
            fallback_chat_template=cfg.model.chat_template,
        ),
    )

    model_name = cfg.model.name
    # Anchor the llm-judge battles against the human arena battles. These are
    # rebuilt from the (revision-pinned) arena, not persisted per run.
    df_arena = arena_anchor_battles(df_arena_all)

    calibrated_temperature = calibrate_pairscore_temperature(
        df_arena,
        df_arena_all,
        enabled=cfg.elo.calibrate_temperature,
        soft_elo=cfg.elo.soft_elo,
        sample_size=cfg.elo.calibration_size,
        rng=rng,
        judge_model=cfg.judge.model,
        judge_model_kwargs=judge_extra_kwargs,
        swap_mode=cfg.judge.swap_mode,
        prompt=resolved_prompt,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        default_temperature=cfg.elo.soft_elo_temperature,
        arena=arena,
        inference_cache=build_judgement_cache(cfg),
    )

    df_llm_judge = build_candidate_battles(
        cfg,
        panel,
        completions,
        judged,
        parser=resolved_prompt.parser,
        effective_temperature=(
            calibrated_temperature
            if calibrated_temperature is not None
            else cfg.elo.soft_elo_temperature
        ),
    )

    df_results = pd.concat([df_llm_judge, df_arena], ignore_index=True)

    metrics = build_metrics(
        protocol.scoring.metrics,
        parameter_overrides_by_metric={
            "bradley_terry": {
                "n_bootstraps": cfg.elo.n_bootstraps,
                "baseline_model": cfg.elo.baseline_model,
                "soft": cfg.elo.soft_elo,
            },
        },
    )
    metric_results = calculate_metrics(
        df_results,
        metrics,
        runtime_by_metric={"bradley_terry": {"rng": rng}},
    )
    report = EloReport(
        arena=arena,
        judge_model=cfg.judge.model,
        metrics=metric_results,
        num_battles=n,
        model_name=model_name,
        sampling_metadata=sampling_metadata,
    )
    results = report.to_dict()
    report.render()
    # ELO artifacts (ratings, battles, bootstrap CSV, metadata) are judge-specific,
    # so key the folder on the judge too — otherwise re-running the same
    # arena/model under a different judge silently overwrites the previous run.
    res_dir = prepare_run_directory(
        cfg,
        Path(cfg.run.result_folder)
        / f"elo-{safe_filename(arena)}-{safe_filename(model_name)}-"
        f"{safe_filename(cfg.judge.model)}",
    )
    result_path = report.save(res_dir / f"results-{safe_filename(model_name)}.json")

    # Persist only the run's own llm-judge battles (a few KB). The human arena
    # anchors are identical across every run, so we do not duplicate them per
    # experiment — recompute ELO by loading this task's pinned battles again and
    # applying arena_anchor_battles(). question_id is the join key back to the
    # arena table / completion cache. battles.parquet keeps pref_hard so both
    # hard and soft ELO can be recomputed.
    battle_cols = [
        "model_a",
        "model_b",
        "winner",
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
    if rating_result is not None and rating_result["bootstrap_ratings"]:
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

    # Reproducibility manifest (git hash, dependency versions, timings) — parity
    # with the other entrypoints, all of which write run-metadata. Best-effort:
    # a metadata-write failure should not sink an already-completed run.
    write_run_metadata_safely(
        output_dir=res_dir,
        entrypoint="judgearena.benchmarks.elo.runner.run_elo",
        run=cfg.model_dump(),
        results=results,
        input_payloads=(
            {"question_id": df_battles["question_id"].tolist()}
            if "question_id" in df_battles.columns
            else None
        ),
        judge_system_prompt=resolved_prompt.system_prompt,
        judge_user_prompt_template=resolved_prompt.user_prompt_template,
        started_at_utc=run_started_at,
    )

    return {
        **results,
        "result_path": str(result_path),
    }
