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
from judgearena.battles import Leaderboard, summarize_bootstrap, write_battles
from judgearena.benchmarks.elo.rating import (
    arena_anchor_battles,
    prefs_to_battle_results,
    sampling_cache_token,
    select_seeded_random_arena_battles,
    winner_to_pref,
)
from judgearena.benchmarks.elo.scoring import ELO_SCORERS
from judgearena.benchmarks.execution import build_generation_kwargs
from judgearena.datasets import load_battles
from judgearena.evaluate import (
    PairScore,
    calibrate_temperature,
    combine_swapped_prefs,
    judge_and_parse_prefs,
    resolve_run_judge_prompt,
)
from judgearena.generate import generate_instructions
from judgearena.log import get_logger
from judgearena.models import build_default_judge_model_kwargs, make_model
from judgearena.tasks.schema import EloProtocol, ResolvedTaskSpec
from judgearena.utils import cache_function_dataframe, compute_pref_summary
from judgearena.utils.eval import PrefSummary, Report

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


class EloReport(Report):
    """Bradley-Terry / Soft-ELO ratings for one focal model against an arena.

    This is the console/``results-*.json`` run report. The narrower per-model
    leaderboard with bootstrap CIs is persisted separately as
    ``elo_ratings.json`` via :class:`judgearena.battles.Leaderboard`.
    """

    arena: str
    """Arena/benchmark the focal model is rated against."""
    judge_model: str
    """LLM judge that scored the battles."""
    summary: PrefSummary
    """Win/loss/tie stats for the focal model's LLM-judged battles."""
    num_battles: int
    """Total battles (LLM-judged + human-anchor)."""
    llm_judged_battles: int
    """Battles the LLM judged for the focal model."""
    human_anchor_battles: int
    """Human-annotated battles anchoring the other arena models."""
    elo_mean: float
    """Focal model's mean ELO across bootstrap samples."""
    elo_std: float
    """Std of the focal model's ELO across bootstrap samples."""
    elo_n_bootstraps: int
    """Bootstrap samples that rated the focal model (≤ n_bootstraps); the n behind elo_mean/elo_std."""
    mae_vs_human: float
    """Mean absolute error of estimated vs human ELO over overlapping models."""
    method: str
    """Rating method label (e.g. "Soft-ELO")."""
    n_bootstraps: int
    """Total bootstrap iterations run."""
    model_name: str
    """Focal model under evaluation."""
    mean_ratings: dict[str, float]
    """Per-model mean ELO across bootstraps."""
    battle_counts: dict[str, int]
    """Per-model battle count."""
    human_elo: dict[str, float]
    """Per-model human-derived ELO (anchors)."""
    bootstrap_ratings: list[dict[str, float]]
    """One model→ELO dict per bootstrap sample."""
    sampling_metadata: dict[str, object]
    """Instruction-sampling parameters for the run."""

    def render(self) -> None:
        s = self.summary
        print(f"\n=== Results for {self.model_name} ===")
        print(
            f"Battles: {self.llm_judged_battles} | Wins: {s.num_wins} | "
            f"Losses: {s.num_losses} | Ties: {s.num_ties}"
        )
        print(f"Win rate: {s.winrate:.2%}")

        print(
            f"\n=== {self.method} Ratings (Bradley-Terry, "
            f"{self.n_bootstraps} bootstraps) ==="
        )
        print(
            f"Estimating {self.method} Ratings with {self.llm_judged_battles} "
            f"LLM-judges for model {self.model_name} and {self.human_anchor_battles} "
            "human annotations for other models. Number of battles is indicated in "
            "parenthesis and confidence intervals are reported by computing ELO on "
            f"{self.n_bootstraps} samples of instructions."
        )

        if not self.mean_ratings:
            print("  Not enough data to compute ELO ratings.")
            return

        # Percentile CIs (not mean ± std): matches the bounds persisted in
        # elo_ratings.json so the console and the saved leaderboard never disagree.
        for e in summarize_bootstrap(
            self.bootstrap_ratings, self.battle_counts, self.model_name
        ):
            suffix = " <-----" if e.model == self.model_name else ""
            print(
                f"  {e.model}  ({e.n_battles}){suffix}: "
                f"{e.rating:.1f} [{e.ci_low:.1f}, {e.ci_high:.1f}]"
            )

        overlap = [
            m for m in self.mean_ratings if m in self.human_elo and m != self.model_name
        ]
        if overlap:
            print(
                f"\n  MAE vs Human-ELO ({len(overlap)} arena models): "
                f"{self.mae_vs_human:.1f}"
            )
        else:
            print("\n  No overlapping arena models to compute MAE.")


def run_elo(cfg: "RunConfig", task: ResolvedTaskSpec | None = None) -> dict:
    """Rate one model against the human battles defined by an ELO task."""
    protocol = task.spec.protocol if task is not None else None
    if not isinstance(protocol, EloProtocol):
        raise ValueError(f"Task {cfg.task!r} does not define an ELO protocol.")
    if cfg.elo is None:
        raise ValueError(f"Task {cfg.task!r} requires ELO runtime settings.")
    arena = protocol.arena
    scorer = ELO_SCORERS[protocol.scoring.adapter]
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
        # Parser side-channel values (e.g. per-criterion scores) become battle
        # columns so downstream scoring can aggregate them.
        values = pd.DataFrame([a.judge_values or {} for a in row_annotations])
        if not values.empty:
            frame = pd.concat([frame, values.set_axis(frame.index)], axis=1)
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

    # Map preferences back to model-name-level battle results.
    model_name = cfg.model.name
    df_llm_judge = prefs_to_battle_results(
        prefs,
        our_model_is_position_a,
        opponent_models,
        model_name,
        judge_model=cfg.judge.model,
        question_ids=question_ids,
    )

    # Normalize prefs so pref < 0.5 always means our model wins, then summarise
    prefs_normalized = pd.Series(
        [
            p if (p is None or is_pos_a) else (1 - p)
            for p, is_pos_a in zip(prefs, our_model_is_position_a, strict=True)
        ]
    )
    summary = compute_pref_summary(prefs_normalized)

    # Anchor the llm-judge battles against the human arena battles. These are
    # rebuilt from the (revision-pinned) arena, not persisted per run.
    df_arena = arena_anchor_battles(df_arena_all)

    df_results = pd.concat([df_llm_judge, df_arena], ignore_index=True)

    # Compute human-only BT ratings as ground-truth reference
    human_elo = scorer.fit(
        df_arena, pref_col="pref_hard", baseline_model=cfg.elo.baseline_model
    )

    # --- Temperature calibration (optional) ---
    # Run the judge on a random subset of human arena battles that already
    # have ground-truth winner labels so we can fit T* via MLE.
    calibrated_temperature: float | None = None
    if cfg.elo.calibrate_temperature:
        if not cfg.elo.soft_elo:
            logger.warning(
                "--calibrate-temperature has no effect with --no-soft-elo; skipping."
            )
        else:
            logger.info("Calibrating PairScore temperature against human annotations.")
            # Sample calibration battles from the already-loaded arena battles.
            # Use the same judge to score them so scores and labels are comparable.
            _cal_n = (
                min(cfg.elo.calibration_size, len(df_arena))
                if cfg.elo.calibration_size is not None
                else len(df_arena)
            )
            # Keep the original df_arena_all index so we can look up the full
            # conversation rows below; reset_index would point at non-existent
            # 0..N labels in df_arena_all.
            cal_battles = df_arena.sample(
                n=_cal_n, random_state=int(rng.integers(0, 2**31))
            )

            cal_instructions = [
                _extract_instruction_text(df_arena_all.loc[i, "conversation_a"][0])
                for i in cal_battles.index
            ]
            cal_completions_a = [
                _extract_instruction_text(df_arena_all.loc[i, "conversation_a"][1])
                for i in cal_battles.index
            ]
            cal_completions_b = [
                _extract_instruction_text(df_arena_all.loc[i, "conversation_b"][1])
                for i in cal_battles.index
            ]

            judge_chat_model_cal = make_model(
                model=cfg.judge.model,
                **judge_extra_kwargs,
            )
            cal_annotations, _, cal_prefs = judge_and_parse_prefs(
                judge_chat_model=judge_chat_model_cal,
                instructions=cal_instructions,
                completions_A=cal_completions_a,
                completions_B=cal_completions_b,
                swap_mode=cfg.judge.swap_mode,
                truncate_input_chars=cfg.generation.truncate_judge_input_chars,
            )

            # Build (delta_s, y) pairs from calibration battles.
            # delta_s = score_A - score_B, extracted exactly as the main run does.
            delta_s_cal = []
            y_cal = []
            for ann, human_winner in zip(
                cal_annotations, cal_battles["winner"].tolist(), strict=True
            ):
                sa, sb = PairScore.parse_raw_scores(ann.judge_completion)
                if sa is None or sb is None:
                    continue
                human_pref = winner_to_pref(human_winner)
                if human_pref is None or human_pref == 0.5:
                    continue  # skip ties and missing
                delta_s_cal.append(sa - sb)
                y_cal.append(1.0 - human_pref)  # pref=0 → A wins → y=1

            if len(delta_s_cal) < 10:
                logger.warning(
                    "Only %d valid calibration pairs (need ≥10); keeping default temperature.",
                    len(delta_s_cal),
                )
            else:
                calibrated_temperature = calibrate_temperature(
                    np.array(delta_s_cal), np.array(y_cal)
                )
                logger.info(
                    "Calibration pairs: %d  T* = %.4f  (default was %s)",
                    len(delta_s_cal),
                    calibrated_temperature,
                    cfg.elo.soft_elo_temperature,
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

        # Rebuild battle results with the re-parsed prefs.
        df_llm_judge = prefs_to_battle_results(
            prefs,
            our_model_is_position_a,
            opponent_models,
            model_name,
            judge_model=cfg.judge.model,
            question_ids=question_ids,
        )
        df_results = pd.concat([df_llm_judge, df_arena], ignore_index=True)

    n_bootstraps = cfg.elo.n_bootstraps
    use_soft = cfg.elo.soft_elo

    n_llm = len(df_llm_judge)
    n_human = len(df_arena)
    method_label = "Soft-ELO" if use_soft else "ELO"

    # Count battles per model across the combined results
    battle_counts: dict[str, int] = {}
    for _, row in df_results.iterrows():
        battle_counts[row["model_a"]] = battle_counts.get(row["model_a"], 0) + 1
        battle_counts[row["model_b"]] = battle_counts.get(row["model_b"], 0) + 1

    pref_col = "pref" if use_soft else "pref_hard"
    bootstrap_ratings: list[dict[str, float]] = []
    for _ in range(n_bootstraps):
        df_sample = df_results.sample(
            n=len(df_results), replace=True, random_state=int(rng.integers(0, 2**31))
        )
        ratings = scorer.fit(
            df_sample, pref_col=pref_col, baseline_model=cfg.elo.baseline_model
        )
        bootstrap_ratings.append(ratings)

    # One percentile-CI summary, reused for the console report, the MAE
    # calculation below, and the persisted elo_ratings.json leaderboard so
    # none of the three can disagree.
    entries: list = []
    mean_ratings: dict[str, float] = {}
    mae = np.nan
    if bootstrap_ratings:
        entries = summarize_bootstrap(bootstrap_ratings, battle_counts, model_name)
        mean_ratings = {e.model: e.rating for e in entries}
        overlap = [m for m in mean_ratings if m in human_elo and m != model_name]
        if overlap:
            abs_errors = [abs(mean_ratings[m] - human_elo[m]) for m in overlap]
            mae = np.mean(abs_errors)

    model_rating_values = [
        rating[model_name] for rating in bootstrap_ratings if model_name in rating
    ]
    elo_mean = (
        float(np.mean(model_rating_values)) if model_rating_values else float("nan")
    )
    elo_std = (
        float(np.std(model_rating_values)) if model_rating_values else float("nan")
    )

    report = EloReport(
        arena=arena,
        judge_model=cfg.judge.model,
        summary=summary,
        num_battles=n,
        llm_judged_battles=n_llm,
        human_anchor_battles=n_human,
        elo_mean=elo_mean,
        elo_std=elo_std,
        elo_n_bootstraps=len(model_rating_values),
        mae_vs_human=mae,
        method=method_label,
        n_bootstraps=n_bootstraps,
        model_name=model_name,
        mean_ratings=mean_ratings,
        battle_counts=battle_counts,
        human_elo=human_elo,
        bootstrap_ratings=bootstrap_ratings,
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
    if bootstrap_ratings:
        pd.DataFrame(bootstrap_ratings).to_csv(
            res_dir / "bootstrap_ratings.csv", index=False
        )
        Leaderboard(
            arena=arena,
            model=model_name,
            judge_model=cfg.judge.model,
            n_bootstraps=n_bootstraps,
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
