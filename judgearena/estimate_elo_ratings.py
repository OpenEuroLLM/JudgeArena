import hashlib
import json
import re
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from judgearena.arenas_utils import _extract_instruction_text, load_arena_dataframe
from judgearena.battles import EloReport, summarize_bootstrap, write_battles
from judgearena.evaluate import (
    PairScore,
    calibrate_temperature,
    combine_swapped_prefs,
    judge_and_parse_prefs,
    resolve_run_judge_prompt,
)
from judgearena.generate import generate_instructions
from judgearena.log import get_logger
from judgearena.repro import _to_jsonable, write_run_metadata
from judgearena.utils import cache_function_dataframe, compute_pref_summary, make_model

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


def _winner_to_pref(winner: str) -> float | None:
    """Convert a hard winner label to a continuous preference value."""
    if winner == "model_a":
        return 0.0
    elif winner == "model_b":
        return 1.0
    elif winner in ("tie", "tie (bothbad)"):
        return 0.5
    return None


def _is_nan_pref(p) -> bool:
    return p is None or (isinstance(p, float) and np.isnan(p))


def fit_bradley_terry(
    df: pd.DataFrame,
    pref_col: str = "pref",
    scale: float = 400,
    base: float = 10,
    init_rating: float = 1000,
    baseline_model: str | None = None,
    baseline_rating: float = 1000,
) -> dict[str, float]:
    """Fit Bradley-Terry ratings via weighted logistic regression.

    Each row in *df* is a battle with columns ``model_a``, ``model_b`` and
    ``pref_col`` ∈ [0, 1] where 0 means A wins, 1 means B wins, 0.5 is a tie.
    Hard win/loss/tie labels are the special case ``pref ∈ {0, 0.5, 1}``.

    The soft cross-entropy for a battle is decomposed into two weighted
    hard-label rows so sklearn's ``LogisticRegression`` can be reused:

        Y=1, weight = (1 − pref) · count   (evidence A wins)
        Y=0, weight =  pref      · count   (evidence B wins)

    Identical ``(model_a, model_b, pref)`` triples are aggregated first so
    the design matrix stays small when prefs are quantised (e.g. human
    arena labels) and untouched when prefs are continuous floats.
    """
    df = df.dropna(subset=[pref_col])
    if df.empty:
        return {}

    grouped = (
        df.groupby(["model_a", "model_b", pref_col]).size().reset_index(name="count")
    )

    all_models = sorted(set(grouped["model_a"]) | set(grouped["model_b"]))
    models = pd.Series(np.arange(len(all_models)), index=all_models)
    p = len(models)

    m_a_idx = grouped["model_a"].map(models).to_numpy()
    m_b_idx = grouped["model_b"].map(models).to_numpy()
    prefs = grouped[pref_col].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n = len(grouped)

    log_base = np.log(base)
    X = np.zeros((2 * n, p))
    top = np.arange(n)
    bot = n + top
    X[top, m_a_idx] = +log_base
    X[top, m_b_idx] = -log_base
    X[bot, m_a_idx] = +log_base
    X[bot, m_b_idx] = -log_base

    Y = np.concatenate([np.ones(n), np.zeros(n)])
    sample_weights = np.concatenate([(1.0 - prefs) * counts, prefs * counts])

    # Keep zero-weight rows so sklearn LR always sees both Y classes — when
    # every pref collapses to 0 or 1 the missing-class rows contribute nothing
    # to the loss but stop the solver from raising on n_classes < 2.
    if sample_weights.sum() == 0:
        return {}

    lr = LogisticRegression(fit_intercept=False, C=1e10, tol=1e-6, max_iter=1000)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = scale * lr.coef_[0] + init_rating

    if baseline_model is not None and baseline_model in models.index:
        elo_scores += baseline_rating - elo_scores[models[baseline_model]]

    return dict(pd.Series(elo_scores, index=models.index))


def _sample_fingerprint(sampled: pd.DataFrame) -> str:
    rows = []
    for index, row in sampled.iterrows():
        rows.append(
            {
                "index": int(index)
                if isinstance(index, int | np.integer)
                else str(index),
                "question_id": str(row["question_id"]),
                "model_a": str(row["model_a"]),
                "model_b": str(row["model_b"]),
            }
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def select_seeded_random_arena_battles(
    df_battles: pd.DataFrame,
    *,
    n_battles: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select a shared random battle panel for outside-model Elo estimation."""
    n = min(n_battles, len(df_battles))
    sampled = df_battles.sample(n=n, random_state=seed, replace=False)
    metadata: dict[str, object] = {
        "sampling_mode": "seeded_random",
        "random_seed": seed,
        "requested_rows": n_battles,
        "sampled_rows": len(sampled),
        "sampled_original_indices": [
            int(index) if isinstance(index, int | np.integer) else str(index)
            for index in sampled.index
        ],
        "sampled_question_ids": [
            str(value) for value in sampled["question_id"].tolist()
        ],
        "sample_fingerprint": _sample_fingerprint(sampled),
    }
    return sampled.reset_index(drop=True), metadata


def _sampling_cache_token(
    sampling_metadata: dict[str, object],
    *,
    n_instructions: int | None,
    n_instructions_per_language: int | None,
) -> str:
    mode = sampling_metadata.get("sampling_mode")
    if mode == "seeded_random":
        return (
            "seeded-random_"
            f"{sampling_metadata['requested_rows']}_"
            f"seed-{sampling_metadata['random_seed']}_"
            f"{str(sampling_metadata['sample_fingerprint'])[:12]}"
        )
    return f"head_{n_instructions}_{n_instructions_per_language}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "model"


def write_elo_result(
    *,
    result_folder: str | Path,
    summary: dict[str, object],
    bootstrap_ratings: list[dict[str, float]],
) -> Path:
    model = str(summary["model_A"])
    arena = str(summary["arena"])
    output_dir = Path(result_folder) / f"elo-{_slugify(arena)}-{_slugify(model)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"results-{_slugify(model)}.json"
    payload = {
        "summary": summary,
        "bootstrap_ratings": bootstrap_ratings,
    }
    path.write_text(json.dumps(_to_jsonable(payload), indent=2) + "\n")
    return path


def _prefs_to_battle_results(
    prefs,
    our_model_is_position_a,
    opponent_models,
    model_name: str,
    *,
    judge_model: str | None = None,
) -> pd.DataFrame:
    """Map per-battle judge prefs into model-name-level battle rows.

    The judge prompt placed our model at position A or B independently per
    battle.  Here we re-orient each row so ``model_a``/``model_b`` carry
    the actual model names and ``pref`` is consistent with that ordering
    (``pref=0`` ⇒ ``model_a`` wins).  ``pref_hard`` is the quantised
    {0, 0.5, 1} version used by the non-soft Bradley-Terry fit.
    """
    records = []
    for pref, is_pos_a, opp in zip(
        prefs, our_model_is_position_a, opponent_models, strict=True
    ):
        if _is_nan_pref(pref) or pref == 0.5:
            winner = "tie"
        elif pref < 0.5:
            winner = "model_a"
        else:
            winner = "model_b"

        if is_pos_a:
            rec = {
                "model_a": model_name,
                "model_b": opp,
                "winner": winner,
                "pref": pref,
            }
        else:
            rec = {
                "model_a": opp,
                "model_b": model_name,
                "winner": winner,
                "pref": None if _is_nan_pref(pref) else pref,
            }
        rec["pref_hard"] = _winner_to_pref(winner)
        records.append(rec)
    df = pd.DataFrame(records)
    df["source"] = "llm-judge"
    df["judge_model"] = judge_model
    return df


def main(cfg: "RunConfig") -> dict:
    assert cfg.elo is not None  # main is dispatched only for elo tasks
    run_started_at = datetime.now(UTC)
    rng = np.random.default_rng(cfg.run.seed)

    # Step 1: Load arena battles
    logger.info("Step 1: Loading battles from %s", cfg.elo.arena)
    df_arena_all = load_arena_dataframe(arena=cfg.elo.arena)

    # Filter by language if specified
    df_battles = df_arena_all
    if cfg.elo.languages:
        df_battles = df_battles[df_battles["lang"].isin(cfg.elo.languages)]

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

    # Only pass extra engine kwargs that are not None
    extra_kwargs = dict(cfg.model.engine_kwargs)
    if cfg.model.max_model_len is not None:
        extra_kwargs["max_model_len"] = cfg.model.max_model_len
    if cfg.model.chat_template is not None:
        extra_kwargs["chat_template"] = cfg.model.chat_template
    use_tqdm = False
    gen_fun = partial(
        generate_instructions,
        truncate_input_chars=cfg.generation.truncate_all_input_chars,
        max_tokens=cfg.model.max_out_tokens,
        use_tqdm=use_tqdm,
        **extra_kwargs,
    )

    def replace_slash(s: str) -> str:
        return s.replace("/", "_")

    languages_str = "-".join(sorted(cfg.elo.languages)) if cfg.elo.languages else "all"
    extra_kwargs_str = (
        "_".join(f"{k}={v}" for k, v in sorted(extra_kwargs.items()))
        if extra_kwargs
        else ""
    )
    sampling_cache_token = _sampling_cache_token(
        sampling_metadata,
        n_instructions=cfg.generation.n_instructions,
        n_instructions_per_language=cfg.elo.n_instructions_per_language,
    )
    cache_suffix = (
        f"{cfg.elo.arena}_{replace_slash(cfg.model.name)}_"
        f"{sampling_cache_token}_"
        f"{languages_str}_{cfg.generation.truncate_all_input_chars}_{cfg.model.max_out_tokens}"
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
    resolved_prompt = resolve_run_judge_prompt(cfg.elo.arena, cfg.judge)

    completions_A = [
        our_completions[i] if our_model_is_position_a[i] else opponent_completions[i]
        for i in range(n)
    ]
    completions_B = [
        opponent_completions[i] if our_model_is_position_a[i] else our_completions[i]
        for i in range(n)
    ]

    judge_extra_kwargs = {}
    if cfg.judge.max_model_len is not None:
        judge_extra_kwargs["max_model_len"] = cfg.judge.max_model_len
    if cfg.model.chat_template is not None:
        judge_extra_kwargs["chat_template"] = cfg.model.chat_template
    judge_extra_kwargs.update(cfg.model.engine_kwargs)
    judge_extra_kwargs.update(cfg.judge.engine_kwargs)

    def run_judge() -> pd.DataFrame:
        judge_chat_model = make_model(
            model=cfg.judge.model,
            max_tokens=cfg.judge.max_out_tokens,
            **judge_extra_kwargs,
        )
        annotations, annotations_reversed, prefs = judge_and_parse_prefs(
            judge_chat_model=judge_chat_model,
            instructions=instructions.tolist(),
            completions_A=completions_A,
            completions_B=completions_B,
            swap_mode=cfg.judge.swap_mode,
            provide_explanation=cfg.judge.provide_explanation,
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
        return pd.DataFrame(
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

    judge_cache_suffix = f"judge_{cache_suffix}"
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

    logger.debug("First judge output:\n%s", df_judge["judge_completion"].iloc[0][:500])

    # Map preferences back to model-name-level battle results.
    model_name = cfg.model.name
    df_llm_judge = _prefs_to_battle_results(
        prefs, our_model_is_position_a, opponent_models, model_name,
        judge_model=cfg.judge.model,
    )

    # Normalize prefs so pref < 0.5 always means our model wins, then summarise
    prefs_normalized = pd.Series(
        [
            p if (p is None or is_pos_a) else (1 - p)
            for p, is_pos_a in zip(prefs, our_model_is_position_a, strict=True)
        ]
    )
    summary = compute_pref_summary(prefs_normalized)
    our_wins = summary["num_wins"]
    our_losses = summary["num_losses"]
    our_ties = summary["num_ties"]
    winrate = summary["winrate"]

    print(f"\n=== Results for {model_name} ===")
    print(
        f"Battles: {len(df_llm_judge)} | Wins: {our_wins} | "
        f"Losses: {our_losses} | Ties: {our_ties}"
    )
    print(f"Win rate: {winrate:.2%}")

    # Combine LLM-judge battles with human-annotated arena battles,
    # keeping only arena models with at least 500 human battles
    df_arena = df_arena_all.loc[:, ["model_a", "model_b", "winner"]].copy()
    human_battle_counts = pd.concat(
        [df_arena["model_a"], df_arena["model_b"]]
    ).value_counts()
    well_represented = set(human_battle_counts[human_battle_counts >= 500].index)
    df_arena = df_arena[
        df_arena["model_a"].isin(well_represented)
        & df_arena["model_b"].isin(well_represented)
    ]
    # Add pref column to arena battles (hard labels → 0.0 / 1.0 / 0.5).
    # Human labels are already hard, so pref_hard == pref.
    df_arena["pref"] = df_arena["winner"].map(_winner_to_pref)
    df_arena["pref_hard"] = df_arena["pref"]
    df_arena["source"] = "human"

    df_results = pd.concat([df_llm_judge, df_arena], ignore_index=True)

    # Compute human-only BT ratings as ground-truth reference
    human_elo = fit_bradley_terry(
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
                max_tokens=cfg.judge.max_out_tokens,
                **judge_extra_kwargs,
            )
            cal_annotations, _, cal_prefs = judge_and_parse_prefs(
                judge_chat_model=judge_chat_model_cal,
                instructions=cal_instructions,
                completions_A=cal_completions_a,
                completions_B=cal_completions_b,
                swap_mode=cfg.judge.swap_mode,
                provide_explanation=cfg.judge.provide_explanation,
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
                human_pref = _winner_to_pref(human_winner)
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
        df_llm_judge = _prefs_to_battle_results(
            prefs, our_model_is_position_a, opponent_models, model_name,
            judge_model=cfg.judge.model,
        )
        df_results = pd.concat([df_llm_judge, df_arena], ignore_index=True)

    n_bootstraps = cfg.elo.n_bootstraps
    use_soft = cfg.elo.soft_elo

    n_llm = len(df_llm_judge)
    n_human = len(df_arena)
    method_label = "Soft-ELO" if use_soft else "ELO"
    print(
        f"\n=== {method_label} Ratings (Bradley-Terry, {n_bootstraps} bootstraps) ==="
    )
    print(
        f"Estimating {method_label} Ratings with {n_llm} LLM-judges for model {model_name} "
        f"and {n_human} human annotations for other models. Number of battles is indicated in parenthesis and "
        f"confidence intervals are reported by computing ELO on {n_bootstraps} samples of instructions."
    )

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
        ratings = fit_bradley_terry(
            df_sample, pref_col=pref_col, baseline_model=cfg.elo.baseline_model
        )
        bootstrap_ratings.append(ratings)

    if bootstrap_ratings:
        all_model_names = sorted(
            set(df_results["model_a"]) | set(df_results["model_b"])
        )
        mean_ratings = {
            m: np.nanmean([r.get(m, np.nan) for r in bootstrap_ratings])
            for m in all_model_names
        }
        for m in sorted(all_model_names, key=lambda x: -mean_ratings[x]):
            vals = [r[m] for r in bootstrap_ratings if m in r]
            suffix = " <-----" if m == model_name else ""
            count = battle_counts.get(m, 0)
            print(f"  {m}  ({count}){suffix}: {np.mean(vals):.1f} ± {np.std(vals):.1f}")

        # MAE vs human-only ELO for overlapping arena models
        overlap = [m for m in all_model_names if m in human_elo and m != model_name]
        if overlap:
            abs_errors = [abs(mean_ratings[m] - human_elo[m]) for m in overlap]
            mae = np.mean(abs_errors)
            print(f"\n  MAE vs Human-ELO ({len(overlap)} arena models): {mae:.1f}")
        else:
            mae = np.nan
            print("\n  No overlapping arena models to compute MAE.")
    else:
        print("  Not enough data to compute ELO ratings.")
        mae = np.nan

    model_rating_values = [
        rating[model_name] for rating in bootstrap_ratings if model_name in rating
    ]
    elo_mean = (
        float(np.mean(model_rating_values)) if model_rating_values else float("nan")
    )
    elo_std = (
        float(np.std(model_rating_values)) if model_rating_values else float("nan")
    )
    result_summary = {
        **summary,
        "arena": cfg.elo.arena,
        "model_A": model_name,
        "judge_model": cfg.judge.model,
        "num_battles": n,
        "llm_judged_battles": n_llm,
        "human_anchor_battles": n_human,
        "sampling_metadata": sampling_metadata,
        "elo_mean": elo_mean,
        "elo_std": elo_std,
        "elo_num_bootstraps": len(model_rating_values),
        "source_battle_counts": battle_counts,
    }
    result_path = write_elo_result(
        result_folder=cfg.run.result_folder,
        summary=result_summary,
        bootstrap_ratings=bootstrap_ratings,
    )

    # Augment write_elo_result's folder with a self-contained battles table
    # (ELO is a pure function of it) plus the percentile-CI leaderboard.
    # battles.parquet carries pref_hard too, so both hard and soft ELO are
    # recomputable from the file alone. question_id is not persisted yet: unlike
    # the position arrays it is not duplicated to 2n for swap_mode="both", so it
    # is left for a follow-up.
    res_dir = result_path.parent
    battle_cols = [
        "model_a", "model_b", "winner", "pref", "pref_hard", "source", "judge_model"
    ]
    write_battles(
        res_dir / "battles.parquet",
        df_results[[c for c in battle_cols if c in df_results.columns]],
    )
    if bootstrap_ratings:
        pd.DataFrame(bootstrap_ratings).to_csv(
            res_dir / "bootstrap_ratings.csv", index=False
        )
        EloReport(
            arena=cfg.elo.arena,
            model=model_name,
            judge_model=cfg.judge.model,
            n_bootstraps=n_bootstraps,
            seed=cfg.run.seed,
            ratings=summarize_bootstrap(bootstrap_ratings, battle_counts, model_name),
        ).write(res_dir / "elo_ratings.json")

    # Reproducibility manifest (git hash, dependency versions, timings) — parity
    # with the other entrypoints, all of which write run-metadata.
    write_run_metadata(
        output_dir=res_dir,
        entrypoint="judgearena.estimate_elo_ratings.main",
        run=cfg.model_dump(),
        results=result_summary,
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
        **result_summary,
        "bootstrap_ratings": bootstrap_ratings,
        "human_elo": human_elo,
        "mae_vs_human": mae,
        "model_name": model_name,
        "result_path": str(result_path),
        "method": method_label,
        "calibrated_temperature": calibrated_temperature,
    }
