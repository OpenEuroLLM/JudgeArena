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
from judgearena.battles import (
    Leaderboard,
    RatingEntry,
    write_battles,
)
from judgearena.benchmarks.elo.calibration import calibrate_pairscore_temperature
from judgearena.benchmarks.elo.rating import (
    arena_anchor_battles,
    prefs_to_battle_results,
    select_seeded_random_arena_battles,
)
from judgearena.benchmarks.execution import (
    build_judgement_cache,
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
from judgearena.models import build_default_judge_model_kwargs, prepare_model
from judgearena.reports import EloReport
from judgearena.tasks.schema import EloProtocol, ResolvedTaskSpec

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


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
            "completion_A": completions.tolist(),
            "completion_B": opponent_completions,
            "model_A": cfg.model.name,
            "model_B": opponent_models,
            "opponent_model": opponent_models,
            "our_model_is_position_a": our_model_is_position_a,
            "question_id": df_battles.get(
                "question_id", df_battles["instruction_index"]
            ).tolist(),
        }
    )
    for prefix in ("completion", "model"):
        comparisons.loc[~our_model_is_position_a, [f"{prefix}_A", f"{prefix}_B"]] = (
            comparisons.loc[
                ~our_model_is_position_a, [f"{prefix}_B", f"{prefix}_A"]
            ].to_numpy()
        )
    resolved_prompt = resolve_run_judge_prompt(cfg.resolved_task, cfg.judge)

    judge_extra_kwargs = build_default_judge_model_kwargs(
        cfg.judge.model,
        cfg.model.engine_kwargs,
        judge_engine_kwargs_override=cfg.judge.model_kwargs(
            fallback_chat_template=cfg.model.chat_template,
        ),
    )

    judge_chat_model = prepare_model(
        model=cfg.judge.model,
        cache=build_judgement_cache(cfg),
        **judge_extra_kwargs,
    )
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

    # Ratings retain the first judge order, including for reversed judgments.
    comparison_rows = comparisons.loc[df_judge["instruction_index"]].reset_index(
        drop=True
    )

    logger.debug("First judge output:\n%s", df_judge["judge_completion"].iloc[0][:500])

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

    # Reparse cached responses at this run's temperature, keeping the selected
    # parser's validation rules.
    if cfg.elo.soft_elo and isinstance(resolved_prompt.parser, PairScore):
        temperature = (
            calibrated_temperature
            if calibrated_temperature is not None
            else cfg.elo.soft_elo_temperature
        )
        reparsed_prefs: list[float] = []
        for completion in df_judge["judge_completion"]:
            parsed = resolved_prompt.parser.parse_result(
                completion, temperature=temperature
            )
            reparsed_prefs.append(float("nan") if parsed is None else parsed.preference)
        reparsed = pd.Series(reparsed_prefs, dtype=float)
        df_judge["pref"] = reparsed.where(
            df_judge["orientation"] != "reversed", 1 - reparsed
        )

    df_llm_judge = prefs_to_battle_results(
        df_judge["pref"],
        comparison_rows["our_model_is_position_a"],
        comparison_rows["opponent_model"],
        model_name,
        judge_model=cfg.judge.model,
        question_ids=comparison_rows["question_id"],
    )
    # Focal metrics consume the original completions in the same frame as prefs.
    df_llm_judge["evaluation_model"] = model_name
    df_llm_judge["completion_a"] = comparison_rows["completion_A"]
    df_llm_judge["completion_b"] = comparison_rows["completion_B"]
    df_llm_judge["instruction_index"] = comparison_rows["question_id"]
    df_llm_judge["orientation"] = df_judge["orientation"]

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
