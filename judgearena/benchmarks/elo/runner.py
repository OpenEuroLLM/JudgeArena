import hashlib
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from judgearena.arenas_utils import _extract_instruction_text
from judgearena.artifacts import (
    prepare_run_directory,
    safe_filename,
    write_run_metadata_safely,
)
from judgearena.battles import Leaderboard, RatingEntry, write_battles
from judgearena.benchmarks.elo.calibration import calibrate_pairscore_temperature
from judgearena.benchmarks.elo.rating import (
    arena_anchor_battles,
    prefs_to_battle_results,
    sampling_cache_token,
    select_seeded_random_arena_battles,
)
from judgearena.benchmarks.execution import build_generation_kwargs
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.datasets import load_battles
from judgearena.evaluate import (
    PairScore,
    combine_swapped_prefs,
    judge_and_parse_prefs,
    resolve_run_judge_prompt,
)
from judgearena.generate import generate_instructions
from judgearena.log import get_logger
from judgearena.models import build_default_judge_model_kwargs, make_model
from judgearena.reports import EloReport
from judgearena.tasks.schema import EloProtocol, ResolvedTaskSpec
from judgearena.utils import cache_function_dataframe

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
    arena = protocol.arena
    run_started_at = datetime.now(UTC)
    rng = np.random.default_rng(cfg.run.seed)

    # Step 1: Load arena battles
    logger.info("Step 1: Loading battles from %s", arena)
    df_arena_all = load_battles(task)

    # Filter by language: a task variant (e.g. elo-lmarena-140k-en) preselects
    # languages; elo.languages narrows further within that selection.
    selected_languages = list(cfg.elo.languages or [])
    if task.selection is not None:
        variant_languages = list(task.selection.values)
        if selected_languages:
            selected_languages = [
                lang for lang in selected_languages if lang in set(variant_languages)
            ]
            if not selected_languages:
                raise ValueError(
                    f"elo.languages {cfg.elo.languages} has no overlap with the "
                    f"languages of task {cfg.task!r} ({variant_languages})."
                )
        else:
            selected_languages = variant_languages

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
            _extract_instruction_text(row["conversation_a"][0])
            for _, row in df_battles.iterrows()
        ],
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
    gen_fun = partial(
        generate_instructions,
        truncate_input_chars=cfg.generation.truncate_all_input_chars,
        use_tqdm=use_tqdm,
        **extra_kwargs,
    )

    def replace_slash(s: str) -> str:
        return s.replace("/", "_")

    languages_str = (
        "-".join(sorted(selected_languages)) if selected_languages else "all"
    )
    extra_kwargs_str = (
        "_".join(f"{k}={v}" for k, v in sorted(extra_kwargs.items()))
        if extra_kwargs
        else ""
    )
    cache_token = sampling_cache_token(
        sampling_metadata,
        n_instructions=cfg.generation.n_instructions,
        n_instructions_per_language=cfg.elo.n_instructions_per_language,
    )
    cache_suffix = (
        f"{arena}_{replace_slash(cfg.model.name)}_"
        f"{cache_token}_"
        f"{languages_str}_{cfg.generation.truncate_all_input_chars}_{extra_kwargs['max_tokens']}"
        + (f"_{extra_kwargs_str}" if extra_kwargs_str else "")
    )
    if len(cache_suffix) > 100:
        cache_hash = hashlib.sha256(cache_suffix.encode()).hexdigest()[:16]
        logger.debug(
            "Cache suffix too long (%d chars), using hash: %s (full: %s)",
            len(cache_suffix),
            cache_hash,
            cache_suffix,
        )
        cache_suffix = cache_hash
    completions_df = cache_function_dataframe(
        lambda: gen_fun(instructions=instructions, model=cfg.model.name),
        ignore_cache=cfg.run.ignore_cache,
        cache_name=f"elo/{cache_suffix}",
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
            _extract_instruction_text(row["conversation_a"][1])
            if use_model_a_as_opponent[i]
            else _extract_instruction_text(row["conversation_b"][1])
        )
        for i, (_, row) in enumerate(df_battles.iterrows())
    ]
    opponent_models = [
        row["model_a"] if use_model_a_as_opponent[i] else row["model_b"]
        for i, (_, row) in enumerate(df_battles.iterrows())
    ]

    our_completions = completions.tolist()
    resolved_prompt = resolve_run_judge_prompt(cfg.task, cfg.judge)

    completions_A = [
        our_completions[i] if our_model_is_position_a[i] else opponent_completions[i]
        for i in range(n)
    ]
    completions_B = [
        opponent_completions[i] if our_model_is_position_a[i] else our_completions[i]
        for i in range(n)
    ]

    judge_extra_kwargs = build_default_judge_model_kwargs(
        cfg.judge.model,
        cfg.model.engine_kwargs,
        judge_engine_kwargs_override=cfg.judge.model_kwargs(
            fallback_chat_template=cfg.model.chat_template,
        ),
    )

    def run_judge() -> pd.DataFrame:
        judge_chat_model = make_model(
            model=cfg.judge.model,
            **judge_extra_kwargs,
        )
        annotations, annotations_reversed, prefs = judge_and_parse_prefs(
            judge_chat_model=judge_chat_model,
            instructions=instructions.tolist(),
            completions_A=completions_A,
            completions_B=completions_B,
            swap_mode=cfg.judge.swap_mode,
            strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
            system_prompt=resolved_prompt.system_prompt,
            user_prompt_template=resolved_prompt.user_prompt_template,
            prompt_preset=resolved_prompt.preset_name,
            parse=resolved_prompt.parser,
            truncate_input_chars=cfg.generation.truncate_judge_input_chars,
            use_tqdm=use_tqdm,
        )
        if annotations_reversed is None:
            row_annotations = list(annotations)
            row_use_model_a = use_model_a_as_opponent
            row_our_pos_a = our_model_is_position_a
            row_opponents = list(opponent_models)
        else:
            # swap_mode="both": dataframe carries 2n rows (AB then BA).
            # Position metadata is duplicated; prefs are already oriented
            # consistently by judge_and_parse_prefs as [pref_AB, 1 - pref_BA].
            row_annotations = list(annotations) + list(annotations_reversed)
            row_use_model_a = np.concatenate(
                [use_model_a_as_opponent, use_model_a_as_opponent]
            )
            row_our_pos_a = np.concatenate(
                [our_model_is_position_a, our_model_is_position_a]
            )
            row_opponents = list(opponent_models) + list(opponent_models)
        frame = pd.DataFrame(
            {
                "judge_completion": [a.judge_completion for a in row_annotations],
                "instruction": [a.instruction for a in row_annotations],
                "completion_A": [a.completion_A for a in row_annotations],
                "completion_B": [a.completion_B for a in row_annotations],
                "pref": prefs,
                "use_model_a_as_opponent": row_use_model_a,
                "our_model_is_position_a": row_our_pos_a,
                "opponent_model": row_opponents,
            }
        )
        return frame

    # Stripping reasoning traces changes the judged text but not the cached
    # completions, so it must be part of the judge cache key. Only append when
    # enabled so prior (non-stripped) runs keep their existing cache hashes.
    judge_cache_suffix = f"judge_{cache_suffix}"
    if cfg.judge.strip_thinking_before_judging:
        judge_cache_suffix += "_stripthinking"
    df_judge = cache_function_dataframe(
        run_judge,
        ignore_cache=cfg.run.ignore_cache,
        cache_name=f"elo/{judge_cache_suffix}",
    )

    # Restore position arrays and prefs from cache (in case loaded from disk)
    use_model_a_as_opponent = df_judge["use_model_a_as_opponent"].to_numpy()
    our_model_is_position_a = df_judge["our_model_is_position_a"].to_numpy()
    opponent_models = df_judge["opponent_model"].tolist()
    prefs = df_judge["pref"].tolist()

    # Instruction-index join key per judged battle, so the saved battles link
    # back to the arena initial table / completion cache without copying text.
    # df_judge repeats the n sampled battles once (AB) or twice (AB then BA for
    # swap_mode="both"), so tile the ids to its actual length.
    if "question_id" in df_battles.columns and len(df_battles):
        qids = df_battles["question_id"].tolist()
        n_rep = (len(df_judge) + len(qids) - 1) // len(qids)
        question_ids = (qids * n_rep)[: len(df_judge)]
    else:
        question_ids = [None] * len(df_judge)

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
    )

    # Build the score parser used for the main evaluation run.
    score_parser = PairScore(
        temperature=calibrated_temperature
        if calibrated_temperature is not None
        else cfg.elo.soft_elo_temperature
    )

    # The prefs cached in df_judge were parsed at the default T=0.3, and the
    # judge cache key ignores temperature, so they cannot reflect
    # --soft-elo-temperature (or a calibrated T*).  Re-parse from the stored
    # judge completions with this run's score_parser so the soft-ELO bootstrap
    # uses the requested temperature.
    if cfg.elo.soft_elo:
        new_prefs_ab = pd.Series(
            [score_parser.parse_model_raw(c) for c in df_judge["judge_completion"]]
        ).apply(lambda x: float("nan") if x is None else x)

        if cfg.judge.swap_mode == "both":
            # df_judge stores AB then BA completions; re-orient the halves the
            # same way run_judge() did.
            n_half = len(df_judge) // 2
            prefs = combine_swapped_prefs(
                new_prefs_ab[:n_half], new_prefs_ab[n_half:]
            ).tolist()
        else:
            prefs = new_prefs_ab.tolist()

    # Map the final canonical preferences to model-name-level battle results.
    df_llm_judge = prefs_to_battle_results(
        prefs,
        our_model_is_position_a,
        opponent_models,
        model_name,
        judge_model=cfg.judge.model,
        question_ids=question_ids,
    )

    # Mark and enrich the rows created for the model under evaluation. Human
    # anchors keep these columns null. Focal metrics can therefore consume the
    # same combined battle table as Bradley-Terry without knowing this runner.
    repeats = max(1, len(df_llm_judge) // max(1, len(our_completions)))
    row_our_completions = (list(our_completions) * repeats)[: len(df_llm_judge)]
    row_opponent_completions = (list(opponent_completions) * repeats)[
        : len(df_llm_judge)
    ]
    focal_is_a = pd.Series(our_model_is_position_a, dtype="bool")
    df_llm_judge["evaluation_model"] = model_name
    df_llm_judge["completion_a"] = pd.Series(row_our_completions).where(
        focal_is_a, row_opponent_completions
    )
    df_llm_judge["completion_b"] = pd.Series(row_opponent_completions).where(
        focal_is_a, row_our_completions
    )
    df_llm_judge["instruction_index"] = question_ids
    if cfg.judge.swap_mode == "both":
        half = len(df_llm_judge) // 2
        df_llm_judge["orientation"] = ["direct"] * half + ["reversed"] * half
    else:
        df_llm_judge["orientation"] = "single"

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
    rating_result = metric_results["bradley_terry"]
    entries = [RatingEntry(**entry) for entry in rating_result["rating_entries"]]

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
    if rating_result["bootstrap_ratings"]:
        pd.DataFrame(rating_result["bootstrap_ratings"]).to_csv(
            res_dir / "bootstrap_ratings.csv", index=False
        )
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
