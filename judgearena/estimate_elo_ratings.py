import hashlib
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from judgearena.arenas_utils import _extract_instruction_text, load_arena_dataframe
from judgearena.cli_common import BaseCliArgs
from judgearena.evaluate import judge_and_parse_prefs, calibrate_temperature, PairScore
from judgearena.generate import generate_instructions
from judgearena.log import get_logger
from judgearena.utils import cache_function_dataframe, compute_pref_summary, make_model

logger = get_logger(__name__)


@dataclass
class CliEloArgs(BaseCliArgs):
    """CLI arguments for the ELO rating estimation entrypoint.

    Note: inheriting from a dataclass (BaseCliArgs) forces every field here to
    have a default value, even for fields like ``arena`` and ``model`` that
    logically should be required.  If this becomes too messy we may want to
    move away from dataclass inheritance.
    """

    arena: str | None = None
    model: str | None = None
    n_instructions_per_language: int | None = None
    languages: list[str] | None = None
    n_bootstraps: int = 20
    seed: int = 0
    baseline_model: str | None = None
    soft_elo: bool = False
    soft_elo_temperature: float = 0.3
    calibrate_temperature: bool = False
    calibration_size: int | None = None
    conformal_alpha: float | None = None
    conformal_min_battles_per_anchor: int = 20


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
        df.groupby(["model_a", "model_b", pref_col])
        .size()
        .reset_index(name="count")
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


def _prefs_to_battle_results(
    prefs,
    our_model_is_position_a,
    opponent_models,
    model_name: str,
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
                "pref": None if _is_nan_pref(pref) else 1.0 - pref,
            }
        rec["pref_hard"] = _winner_to_pref(winner)
        records.append(rec)
    return pd.DataFrame(records)


def _compute_conformal_qhat(
    cal_annotations: list,
    cal_battles: pd.DataFrame,
    df_arena: pd.DataFrame,
    score_parser: "PairScore",
    human_elo: dict[str, float],
    alpha: float,
    min_battles_per_anchor: int,
    baseline_model: str | None = None,
) -> dict:
    """Conformal quantile from leave-one-anchor-out residuals.

    Re-parses the judge annotations already collected for temperature
    calibration to build (model_a, model_b, human_pref, judge_pref) rows,
    keeps anchors that appear in at least ``min_battles_per_anchor`` of
    those rows, then for each surviving anchor refits BT on the human pool
    excluding that anchor plus the anchor's judge-scored battles. The
    residuals (human_elo − judge_elo) feed the standard split-conformal
    quantile  q̂_α = ⌈(K+1)(1−α)⌉ / K -th order statistic of |residual|.
    """
    rows = []
    for ann, (_, battle) in zip(cal_annotations, cal_battles.iterrows(), strict=True):
        pref_j = score_parser.parse_model_raw(ann.judge_completion)
        if pref_j is None:
            continue
        rows.append(
            {
                "model_a": battle["model_a"],
                "model_b": battle["model_b"],
                "human_pref": _winner_to_pref(battle["winner"]),
                "judge_pref": float(pref_j),
            }
        )
    cal_df = pd.DataFrame(rows).dropna(subset=["human_pref"])
    if cal_df.empty:
        return {"qhat": None, "n_anchors": 0, "residuals": {}, "eligible_anchors": []}

    appearances = pd.concat([cal_df["model_a"], cal_df["model_b"]]).value_counts()
    eligible_anchors = [
        m
        for m, c in appearances.items()
        if c >= min_battles_per_anchor and m in human_elo
    ]
    if len(eligible_anchors) < 8:
        logger.warning(
            "Conformal: only %d anchors with >=%d judge-scored calibration battles "
            "and a human Elo reference (need >=8); skipping interval.",
            len(eligible_anchors),
            min_battles_per_anchor,
        )
        return {
            "qhat": None,
            "n_anchors": len(eligible_anchors),
            "residuals": {},
            "eligible_anchors": eligible_anchors,
        }

    human_pool = df_arena[["model_a", "model_b", "pref_hard"]].rename(
        columns={"pref_hard": "pref"}
    )
    residuals: dict[str, float] = {}
    for anchor in eligible_anchors:
        anchor_human = human_pool[
            (human_pool["model_a"] != anchor) & (human_pool["model_b"] != anchor)
        ]
        anchor_judge = cal_df[
            (cal_df["model_a"] == anchor) | (cal_df["model_b"] == anchor)
        ][["model_a", "model_b", "judge_pref"]].rename(columns={"judge_pref": "pref"})
        combined = pd.concat([anchor_human, anchor_judge], ignore_index=True)
        ratings = fit_bradley_terry(
            combined, pref_col="pref", baseline_model=baseline_model
        )
        if anchor in ratings:
            residuals[anchor] = float(human_elo[anchor] - ratings[anchor])

    K = len(residuals)
    if K < 8:
        return {
            "qhat": None,
            "n_anchors": K,
            "residuals": residuals,
            "eligible_anchors": eligible_anchors,
        }

    abs_res = np.abs(list(residuals.values()))
    level = min(np.ceil((K + 1) * (1 - alpha)) / K, 1.0)
    qhat = float(np.quantile(abs_res, level, method="higher"))
    return {
        "qhat": qhat,
        "n_anchors": K,
        "residuals": residuals,
        "eligible_anchors": eligible_anchors,
    }


def main(args: CliEloArgs) -> dict:
    rng = np.random.default_rng(args.seed)

    # Step 1: Load arena battles
    logger.info("Step 1: Loading battles from %s", args.arena)
    df_arena_all = load_arena_dataframe(arena=args.arena)

    # Filter by language if specified
    df_battles = df_arena_all
    if args.languages:
        df_battles = df_battles[df_battles["lang"].isin(args.languages)]

    # Keep at most n_instructions_per_language per language
    if args.n_instructions_per_language is not None:
        df_battles = (
            df_battles.groupby("lang")
            .head(args.n_instructions_per_language)
            .reset_index(drop=True)
        )

    # Keep at most n_instructions total (subset used for LLM-judge evaluation)
    if args.n_instructions is not None:
        df_battles = df_battles.head(args.n_instructions)

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
    logger.info("Step 2: Generating completions with %s", args.model)

    # Only pass extra engine kwargs that are not None
    extra_kwargs = dict(args.engine_kwargs)
    if args.max_model_len is not None:
        extra_kwargs["max_model_len"] = args.max_model_len
    if args.chat_template is not None:
        extra_kwargs["chat_template"] = args.chat_template
    use_tqdm = False
    gen_fun = partial(
        generate_instructions,
        truncate_input_chars=args.truncate_all_input_chars,
        max_tokens=args.max_out_tokens_models,
        use_tqdm=use_tqdm,
        **extra_kwargs,
    )

    def replace_slash(s: str) -> str:
        return s.replace("/", "_")

    languages_str = "-".join(sorted(args.languages)) if args.languages else "all"
    extra_kwargs_str = (
        "_".join(f"{k}={v}" for k, v in sorted(extra_kwargs.items()))
        if extra_kwargs
        else ""
    )
    cache_suffix = (
        f"{args.arena}_{replace_slash(args.model)}_"
        f"{args.n_instructions}_{args.n_instructions_per_language}_"
        f"{languages_str}_{args.truncate_all_input_chars}_{args.max_out_tokens_models}"
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
        lambda: gen_fun(instructions=instructions, model=args.model),
        ignore_cache=args.ignore_cache,
        cache_name=f"elo/{cache_suffix}",
    ).set_index("instruction_index")
    completions = completions_df.loc[:, "completion"]

    logger.debug("First completion:\n%s", completions.iloc[0])

    # Step 3: Judge evaluation against randomly picked arena opponents
    logger.info("Step 3: Judge evaluation with %s", args.judge_model)

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

    completions_A = [
        our_completions[i] if our_model_is_position_a[i] else opponent_completions[i]
        for i in range(n)
    ]
    completions_B = [
        opponent_completions[i] if our_model_is_position_a[i] else our_completions[i]
        for i in range(n)
    ]

    judge_extra_kwargs = {}
    if args.max_model_len is not None:
        judge_extra_kwargs["max_model_len"] = args.max_model_len
    if args.chat_template is not None:
        judge_extra_kwargs["chat_template"] = args.chat_template

    def run_judge() -> pd.DataFrame:
        judge_chat_model = make_model(
            model=args.judge_model,
            max_tokens=args.max_out_tokens_judge,
            **judge_extra_kwargs,
        )
        annotations, annotations_reversed, prefs = judge_and_parse_prefs(
            judge_chat_model=judge_chat_model,
            instructions=instructions.tolist(),
            completions_A=completions_A,
            completions_B=completions_B,
            swap_mode=args.swap_mode,
            provide_explanation=args.provide_explanation,
            truncate_input_chars=args.truncate_all_input_chars,
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
        ignore_cache=args.ignore_cache,
        cache_name=f"elo/{judge_cache_suffix}",
    )

    # Restore position arrays and prefs from cache (in case loaded from disk)
    use_model_a_as_opponent = df_judge["use_model_a_as_opponent"].to_numpy()
    our_model_is_position_a = df_judge["our_model_is_position_a"].to_numpy()
    opponent_models = df_judge["opponent_model"].tolist()
    prefs = df_judge["pref"].tolist()

    logger.debug("First judge output:\n%s", df_judge["judge_completion"].iloc[0][:500])

    # Map preferences back to model-name-level battle results.
    model_name = args.model
    df_llm_judge = _prefs_to_battle_results(
        prefs, our_model_is_position_a, opponent_models, model_name
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

    df_results = pd.concat([df_llm_judge, df_arena], ignore_index=True)

    # Compute human-only BT ratings as ground-truth reference
    human_elo = fit_bradley_terry(
        df_arena, pref_col="pref_hard", baseline_model=args.baseline_model
    )

    # --- Temperature calibration (optional) ---
    # Run the judge on a random subset of human arena battles that already
    # have ground-truth winner labels so we can fit T* via MLE.
    calibrated_temperature: float | None = None
    cal_annotations: list | None = None
    cal_battles: pd.DataFrame | None = None
    if args.calibrate_temperature:
        if not args.soft_elo:
            logger.warning(
                "--calibrate-temperature has no effect without --soft-elo; skipping."
            )
        else:
            logger.info("Calibrating PairScore temperature against human annotations.")
            # Sample calibration battles from the already-loaded arena battles.
            # Use the same judge to score them so scores and labels are comparable.
            _cal_n = (
                min(args.calibration_size, len(df_arena))
                if args.calibration_size is not None
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
                model=args.judge_model,
                max_tokens=args.max_out_tokens_judge,
                **judge_extra_kwargs,
            )
            cal_annotations, _, cal_prefs = judge_and_parse_prefs(
                judge_chat_model=judge_chat_model_cal,
                instructions=cal_instructions,
                completions_A=cal_completions_a,
                completions_B=cal_completions_b,
                swap_mode=args.swap_mode,
                truncate_input_chars=args.truncate_all_input_chars,
            )

            # Build (delta_s, y) pairs from calibration battles.
            # delta_s = score_A - score_B (raw, using default T=1 to extract scores)
            raw_parser = PairScore(temperature=1.0)
            delta_s_cal = []
            y_cal = []
            for ann, human_winner in zip(
                cal_annotations, cal_battles["winner"].tolist(), strict=True
            ):
                sa = raw_parser.get_regexp_match(
                    ann.judge_completion.lower(), r'score.*?a[":\s*\n]*(-?\d+)'
                )
                sb = raw_parser.get_regexp_match(
                    ann.judge_completion.lower(), r'score.*?b[":\s*\n]*(-?\d+)'
                )
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
                    args.soft_elo_temperature,
                )

    # Build the score parser used for the main evaluation run.
    score_parser = PairScore(
        temperature=calibrated_temperature if calibrated_temperature is not None else args.soft_elo_temperature
    )

    # If we calibrated the temperature, the prefs stored in df_judge were
    # computed with the default T=0.3.  Re-parse them with the new parser so
    # the soft-ELO bootstrap uses calibrated preferences.
    if calibrated_temperature is not None:
        new_prefs_ab = pd.Series(
            [score_parser.parse_model_raw(c) for c in df_judge["judge_completion"]]
        )
        prefs = new_prefs_ab.tolist()

        def _none_to_nan(x):
            return float("nan") if x is None else x

        if args.swap_mode == "both":
            # df_judge contains AB and BA annotations interleaved; the original
            # run_judge() already combined them — we just need to re-parse the
            # stored completions in the same order.
            n_half = len(df_judge) // 2
            prefs_ab = new_prefs_ab[:n_half].apply(_none_to_nan)
            prefs_ba = new_prefs_ab[n_half:].apply(_none_to_nan).reset_index(drop=True)
            prefs = pd.concat([prefs_ab, 1 - prefs_ba]).reset_index(drop=True).tolist()

        # Rebuild battle_results with calibrated prefs
        df_llm_judge = _prefs_to_battle_results(
            prefs, our_model_is_position_a, opponent_models, model_name
        )
        df_results = pd.concat([df_llm_judge, df_arena], ignore_index=True)

    n_bootstraps = args.n_bootstraps
    use_soft = args.soft_elo

    n_llm = len(df_llm_judge)
    n_human = len(df_arena)
    method_label = "Soft-ELO" if use_soft else "ELO"
    print(f"\n=== {method_label} Ratings (Bradley-Terry, {n_bootstraps} bootstraps) ===")
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
            df_sample, pref_col=pref_col, baseline_model=args.baseline_model
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
            print(
                f"\n  MAE vs Human-ELO ({len(overlap)} arena models): {mae:.1f}"
            )
        else:
            mae = np.nan
            print("\n  No overlapping arena models to compute MAE.")
    else:
        print("  Not enough data to compute ELO ratings.")
        mae = np.nan
        mean_ratings = {}

    # Conformal interval (optional)
    conformal_result: dict | None = None
    if args.conformal_alpha is not None:
        if not (0.0 < args.conformal_alpha < 1.0):
            logger.warning(
                "--conformal-alpha=%s outside (0, 1); skipping interval.",
                args.conformal_alpha,
            )
        elif cal_annotations is None or cal_battles is None:
            logger.warning(
                "--conformal-alpha requires --calibrate-temperature to produce "
                "judge-scored human battles; skipping interval."
            )
        else:
            conformal_result = _compute_conformal_qhat(
                cal_annotations=cal_annotations,
                cal_battles=cal_battles,
                df_arena=df_arena,
                score_parser=score_parser,
                human_elo=human_elo,
                alpha=args.conformal_alpha,
                min_battles_per_anchor=args.conformal_min_battles_per_anchor,
                baseline_model=args.baseline_model,
            )
            qhat = conformal_result["qhat"]
            if qhat is not None and model_name in mean_ratings:
                point = mean_ratings[model_name]
                lo, hi = point - qhat, point + qhat
                conformal_result["point_estimate"] = float(point)
                conformal_result["interval_lo"] = float(lo)
                conformal_result["interval_hi"] = float(hi)
                print(
                    f"\n=== Conformal Interval (α={args.conformal_alpha:.2f}, "
                    f"K={conformal_result['n_anchors']} anchors) ==="
                )
                print(f"  q̂ = {qhat:.1f} Elo")
                print(f"  {model_name}: {point:.1f} ∈ [{lo:.1f}, {hi:.1f}]")

    return {
        **summary,
        "bootstrap_ratings": bootstrap_ratings,
        "human_elo": human_elo,
        "mae_vs_human": mae,
        "model_name": model_name,
        "method": method_label,
        "calibrated_temperature": calibrated_temperature,
        "conformal": conformal_result,
    }
