from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from judgearena.artifacts import (
    prepare_run_directory,
    run_logging,
    safe_filename,
    write_run_metadata_safely,
)
from judgearena.battles import (
    Leaderboard,
    RatingEntry,
    judged_battles,
    swap_comparison_slots,
    write_battles,
)
from judgearena.benchmarks.elo.calibration import calibrate_pairscore_temperature
from judgearena.benchmarks.elo.rating import (
    arena_anchor_battles,
    select_seeded_random_arena_battles,
)
from judgearena.benchmarks.execution import (
    build_judge,
    load_or_generate_completions,
)
from judgearena.benchmarks.pairwise.judging import judge_comparisons
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.datasets import load_task_data, resolve_dataset_filters
from judgearena.datasets.data import PreferenceData
from judgearena.evaluate import (
    PairScore,
    resolve_run_judge_prompt,
)
from judgearena.generate import generate_instructions
from judgearena.log import get_logger
from judgearena.prompts.parsing import score_preference
from judgearena.reports import EloReport
from judgearena.tasks.schema import EloProtocol, ResolvedTaskSpec

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


@run_logging()
def run_elo(cfg: "RunConfig", task: ResolvedTaskSpec | None = None) -> dict:
    """Rate one model against the human battles defined by an ELO task."""
    cfg = cfg.for_task(task)
    task = cfg.resolved_task
    protocol = task.spec.protocol
    if not isinstance(protocol, EloProtocol):
        raise ValueError(f"Task {cfg.task!r} does not define an ELO protocol.")
    if cfg.elo is None:
        raise ValueError(f"Task {cfg.task!r} requires ELO runtime settings.")
    cfg.elo = cfg.elo.resolve(protocol.scoring)
    arena = protocol.arena
    run_started_at = datetime.now(UTC)
    rng = np.random.default_rng(cfg.run.seed)
    # ELO artifacts (ratings, battles, bootstrap CSV, metadata) are judge-specific,
    # so key the folder on the judge too — otherwise re-running the same
    # arena/model under a different judge silently overwrites the previous run.
    res_dir = prepare_run_directory(
        cfg,
        Path(cfg.run.result_folder)
        / f"elo-{safe_filename(arena)}-{safe_filename(cfg.model.name)}-"
        f"{safe_filename(cfg.judge.model)}",
    )

    # Step 1: Load arena battles
    logger.info("Step 1: Loading battles from %s", arena)
    task_data = load_task_data(task)
    if not isinstance(task_data, PreferenceData):
        raise ValueError("Anchored Elo requires PreferenceData with human labels.")
    df_arena_all = task_data.human_battles()

    filters = resolve_dataset_filters(
        task,
        cfg.run.filters,
        {"lang": cfg.elo.languages} if cfg.elo.languages else {},
    )
    # Selection narrows evaluated rows; human anchors retain the original arena pool.
    df_battles = task_data.select(filters).human_battles()

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

    instructions = pd.Series(
        df_battles["instruction"].tolist(),
        index=df_battles.get(
            "instruction_id", df_battles["instruction_index"].astype(str)
        ),
        name="instruction",
    )
    logger.debug("First instruction:\n%s", instructions.iloc[0][:300])

    # Step 2: Generate completions for the model under evaluation
    logger.info("Step 2: Generating completions with %s", cfg.model.name)

    use_tqdm = False
    completions = load_or_generate_completions(
        cfg,
        cfg.model.name,
        instructions,
        task_data.model_completion(
            cfg.model.name, instruction_ids=pd.Index(df_battles["instruction_index"])
        ),
        role="A",
        generate=generate_instructions,
        use_tqdm=use_tqdm,
    )

    logger.debug("First completion:\n%s", completions.iloc[0])

    # Step 3: Judge evaluation against randomly picked arena opponents
    logger.info("Step 3: Judge evaluation with %s", cfg.judge.model)

    # For each battle, randomly pick opponent: model_a or model_b from the arena
    use_model_a_as_opponent = rng.choice([True, False], size=n)
    # Randomly decide if our model is in position A or B for the judge
    our_model_is_position_a = rng.choice([True, False], size=n)

    opponent_completions = (
        df_battles["completion_a"]
        .where(use_model_a_as_opponent, df_battles["completion_b"])
        .tolist()
    )
    opponent_models = (
        df_battles["model_a"]
        .where(use_model_a_as_opponent, df_battles["model_b"])
        .tolist()
    )

    comparisons = pd.DataFrame(
        {
            "instruction_id": instructions.index,
            "cache_orientation": np.where(
                our_model_is_position_a, "direct", "reversed"
            ),
            "instruction": instructions.tolist(),
            "completion_a": completions.tolist(),
            "completion_b": opponent_completions,
            "model_a": cfg.model.name,
            "model_b": opponent_models,
            "opponent_model": opponent_models,
            "our_model_is_position_a": our_model_is_position_a,
            "question_id": df_battles.get(
                "question_id", df_battles["instruction_index"]
            ).tolist(),
        }
    )
    comparisons = swap_comparison_slots(comparisons, ~our_model_is_position_a)
    resolved_prompt = resolve_run_judge_prompt(cfg.resolved_task, cfg.judge)

    judge_chat_model = build_judge(cfg)
    df_judge = judge_comparisons(
        judge_chat_model,
        comparisons,
        [(resolved_prompt, comparisons.index)],
        # Elo already chose its initial slots using the run RNG.
        swap_mode="both" if cfg.judge.swap_mode == "both" else "fixed",
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        use_tqdm=use_tqdm,
    )
    logger.debug("First judge output:\n%s", df_judge["judge_completion"].iloc[0][:500])

    model_name = cfg.model.name
    # Anchor the llm-judge battles against the human arena battles. These are
    # rebuilt from the (revision-pinned) arena, not persisted per run.
    df_arena = arena_anchor_battles(df_arena_all)

    calibrated_temperature = None
    if cfg.elo.calibrate_temperature:
        if not cfg.elo.soft_elo:
            logger.warning(
                "--calibrate-temperature has no effect with --no-soft-elo; skipping."
            )
        else:
            calibrated_temperature = calibrate_pairscore_temperature(
                df_arena_all.loc[df_arena.index],
                sample_size=cfg.elo.calibration_size,
                rng=rng,
                judge_chat_model=judge_chat_model,
                swap_mode=cfg.judge.swap_mode,
                prompt=resolved_prompt,
                truncate_input_chars=cfg.generation.truncate_judge_input_chars,
                default_temperature=cfg.elo.soft_elo_temperature,
            )

    # Apply this run's temperature to the selected parser's validated scores.
    if cfg.elo.soft_elo and isinstance(resolved_prompt.parser, PairScore):
        temperature = (
            calibrated_temperature
            if calibrated_temperature is not None
            else cfg.elo.soft_elo_temperature
        )
        adjusted_prefs: list[float | None] = []
        for parsed in df_judge["parsed"]:
            scores = {} if parsed is None else parsed["scores"]
            adjusted_prefs.append(
                score_preference(
                    scores.get("A"), scores.get("B"), temperature=temperature
                )
            )
        adjusted = pd.Series(adjusted_prefs, index=df_judge.index, dtype=float)
        df_judge["pref"] = adjusted.where(
            df_judge["orientation"] != "reversed", 1 - adjusted
        )

    df_llm_judge = judged_battles(
        comparisons,
        df_judge,
        evaluation_model=model_name,
        judge=cfg.judge.model,
        judge_temperature=cfg.judge.temperature,
        judge_max_out_tokens=cfg.judge.max_out_tokens,
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
        task=cfg.resolved_task,
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
