from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from judgearena.arenas_utils import _extract_instruction_text, load_arena_dataframe
from judgearena.cli_common import BaseCliArgs
from judgearena.evaluate import PairScore, judge_and_parse_prefs
from judgearena.generate import generate_instructions
from judgearena.log import get_logger
from judgearena.utils import cache_function_dataframe, compute_pref_summary, make_model

if TYPE_CHECKING:
    pass

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
    store_root: str | None = (
        None  # root dir of the SQLite store; enables caching if set
    )


def compute_bradley_terry(
    df: pd.DataFrame,
    winner_col: str,
    scale: float = 400,
    base: float = 10,
    init_rating: float = 1000,
    baseline_model: str | None = None,
    baseline_rating: float = 1000,
) -> dict[str, float]:
    """
    Compute Bradley-Terry ratings using MLE (logistic regression).

    This method fits a Bradley-Terry model to pairwise comparison data using
    maximum likelihood estimation via logistic regression.

    Args:
        df: DataFrame with columns 'model_a', 'model_b', and the winner column
        winner_col: Name of the column containing the winner
        scale: Scale factor for ELO conversion (default 400)
        base: Base for logarithm in ELO formula (default 10)
        init_rating: Initial rating offset (default 1000)
        baseline_model: Model to anchor at baseline_rating
        baseline_rating: Rating to assign to the baseline model

    Returns:
        Dictionary mapping model names to their Bradley-Terry ratings
    """
    # Get all unique models
    all_models = sorted(set(df["model_a"].unique()) | set(df["model_b"].unique()))

    # Create pivot tables for wins
    ptbl_a_win = pd.pivot_table(
        df[df[winner_col] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    ptbl_b_win = pd.pivot_table(
        df[df[winner_col] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Handle ties
    if sum(df[winner_col].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=all_models, columns=all_models)
    else:
        ptbl_tie = pd.pivot_table(
            df[df[winner_col].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie.reindex(index=all_models, columns=all_models, fill_value=0)
        ptbl_tie = ptbl_tie + ptbl_tie.T

    # Reindex all pivot tables to have consistent dimensions
    ptbl_a_win = ptbl_a_win.reindex(index=all_models, columns=all_models, fill_value=0)
    ptbl_b_win = ptbl_b_win.reindex(index=all_models, columns=all_models, fill_value=0)

    # Combined win matrix (ties count as 0.5 for each)
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # Skip if nan or no battles between this pair
            w_ab = ptbl_win.loc[m_a, m_b]
            w_ba = ptbl_win.loc[m_b, m_a]
            if np.isnan(w_ab) or np.isnan(w_ba):
                continue
            if w_ab == 0 and w_ba == 0:
                continue
            X[cur_row, models[m_a]] = +np.log(base)
            X[cur_row, models[m_b]] = -np.log(base)
            Y[cur_row] = 1.0
            sample_weights.append(w_ab)

            X[cur_row + 1, models[m_a]] = np.log(base)
            X[cur_row + 1, models[m_b]] = -np.log(base)
            Y[cur_row + 1] = 0.0
            sample_weights.append(w_ba)
            cur_row += 2

    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, C=1e10, tol=1e-6, max_iter=1000)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = scale * lr.coef_[0] + init_rating

    # Normalize to baseline model if specified
    if baseline_model is not None and baseline_model in models.index:
        elo_scores += baseline_rating - elo_scores[models[baseline_model]]

    return dict(pd.Series(elo_scores, index=models.index))


def _store_folder(store_root: str, kind: str, task: str, model_spec: str) -> Path:
    provider, model_path = model_spec.split("/", 1)
    model_name = model_path.replace("/", "--")
    return Path(store_root) / kind / task / model_name / provider


def main(args: CliEloArgs) -> dict:
    from judgearena.store_sqlite import SQLiteCompletionStore, SQLiteJudgementStore

    if args.store_root is not None:
        comp_folder = _store_folder(
            args.store_root, "completions", args.arena, args.model
        )
        completion_store = SQLiteCompletionStore(comp_folder / "completions.db")
        judge_folder = _store_folder(
            args.store_root, "judgements", args.arena, args.judge_model
        )
        judgement_store = SQLiteJudgementStore(judge_folder / "judgements.db")
        logger.info("Using SQLite store at %s", args.store_root)
    else:
        completion_store = None
        judgement_store = None

    rng = np.random.default_rng(args.seed)

    # Step 1: Load arena battles
    logger.info("Step 1: Loading battles from %s", args.arena)
    df_arena_all = load_arena_dataframe(arena=args.arena)

    # Filter by language if specified
    df_battles = df_arena_all
    if args.languages:
        if args.languages in ["ALL", ["ALL"]]:
            args.languages = [
                "hr",
                "cs",
                "en",
                "ro",
                "it",
                "th",
                "eo",
                "fi",
                "hu",
                "uk",
                "ko",
                "el",
                "fa",
                "ja",
                "sv",
                "nl",
                "he",
                "zh",
                "bn",
                "tr",
                "id",
                "sl",
                "de",
                "es",
                "ca",
                "pt",
                "ru",
                "vi",
                "no",
                "fr",
                "pl",
                "sr",
                "ar",
            ]
        print(f"filtering with languages: {args.languages}")
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
        completion_store=completion_store,
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

    # Build per-row model_A / model_B now that positions are fixed
    model_A_per_row = [
        args.model if our_model_is_position_a[i] else opponent_models[i]
        for i in range(n)
    ]
    model_B_per_row = [
        opponent_models[i] if our_model_is_position_a[i] else args.model
        for i in range(n)
    ]

    judge_extra_kwargs = {}
    if args.max_model_len is not None:
        judge_extra_kwargs["max_model_len"] = args.max_model_len
    if args.chat_template is not None:
        judge_extra_kwargs["chat_template"] = args.chat_template

    def run_judge() -> pd.DataFrame:
        # Determine which indices still need judging
        all_indices = list(range(n))
        if judgement_store is not None:
            pairs = list(
                zip(all_indices, model_A_per_row, model_B_per_row, strict=True)
            )
            cached_df = judgement_store.query(model=args.model)
            cached_keys = (
                set(
                    zip(
                        cached_df["instruction_index"].astype(int),
                        cached_df["model_A"],
                        cached_df["model_B"],
                        strict=True,
                    )
                )
                if not cached_df.empty
                else set()
            )
            missing_idx = [i for i, mA, mB in pairs if (i, mA, mB) not in cached_keys]
            logger.info(
                "Judgement store: %d cached, %d to judge.",
                n - len(missing_idx),
                len(missing_idx),
            )
        else:
            missing_idx = all_indices
            cached_df = pd.DataFrame()

        new_df = pd.DataFrame()
        if missing_idx:
            judge_chat_model = make_model(
                model=args.judge_model,
                max_tokens=args.max_out_tokens_judge,
                **judge_extra_kwargs,
            )
            annotations, _, prefs = judge_and_parse_prefs(
                judge_chat_model=judge_chat_model,
                instructions=[instructions[i] for i in missing_idx],
                completions_A=[completions_A[i] for i in missing_idx],
                completions_B=[completions_B[i] for i in missing_idx],
                swap_mode=args.swap_mode,
                provide_explanation=args.provide_explanation,
                truncate_input_chars=args.truncate_all_input_chars,
                use_tqdm=use_tqdm,
            )
            new_df = pd.DataFrame(
                {
                    "judge_completion": [a.judge_completion for a in annotations],
                    "instruction": [a.instruction for a in annotations],
                    "completion_A": [a.completion_A for a in annotations],
                    "completion_B": [a.completion_B for a in annotations],
                    "pref": list(prefs),
                    "use_model_a_as_opponent": use_model_a_as_opponent[missing_idx],
                    "our_model_is_position_a": our_model_is_position_a[missing_idx],
                    "opponent_model": [opponent_models[i] for i in missing_idx],
                    "instruction_index": missing_idx,
                    "model_A": [model_A_per_row[i] for i in missing_idx],
                    "model_B": [model_B_per_row[i] for i in missing_idx],
                }
            )
            if judgement_store is not None:
                judgement_store.save(
                    new_df.rename(columns={"judge_completion": "judge_output"}),
                    pushed_by="judgearena",
                )

        if judgement_store is not None and not cached_df.empty:
            # Reconstruct full df by merging cached rows back in
            score_parser = PairScore()
            cached_df = cached_df.copy()
            cached_df["pref"] = cached_df["judge_output"].apply(
                score_parser.parse_model_raw
            )
            cached_df = cached_df.rename(columns={"judge_output": "judge_completion"})
            cached_df["our_model_is_position_a"] = cached_df["model_A"] == args.model
            cached_df["opponent_model"] = cached_df.apply(
                lambda r: (
                    r["model_B"] if r["our_model_is_position_a"] else r["model_A"]
                ),
                axis=1,
            )
            cached_df["use_model_a_as_opponent"] = [
                df_battles.iloc[int(idx)]["model_a"] == opp
                for idx, opp in zip(
                    cached_df["instruction_index"],
                    cached_df["opponent_model"],
                    strict=True,
                )
            ]
            full_df = pd.concat([new_df, cached_df], ignore_index=True)
            return full_df.sort_values("instruction_index").reset_index(drop=True)

        return new_df

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

    # Map preferences back to model-name-level battle results
    model_name = args.model
    battle_results = []
    for pref, is_pos_a, opp_model in zip(
        prefs, our_model_is_position_a, opponent_models, strict=True
    ):
        if pref is None or pref == 0.5:
            winner = "tie"
        elif pref < 0.5:
            winner = "model_a"
        else:
            winner = "model_b"

        if is_pos_a:
            battle_results.append(
                {"model_a": model_name, "model_b": opp_model, "winner": winner}
            )
        else:
            battle_results.append(
                {"model_a": opp_model, "model_b": model_name, "winner": winner}
            )

    # LLM-judge battle results for our model
    df_llm_judge = pd.DataFrame(battle_results)

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
    print(f"Battles: {n} | Wins: {our_wins} | Losses: {our_losses} | Ties: {our_ties}")
    print(f"Win rate: {winrate:.2%}")

    # Combine LLM-judge battles with human-annotated arena battles,
    # keeping only arena models with at least 500 human battles
    df_arena = df_arena_all.loc[:, ["model_a", "model_b", "winner"]]
    human_battle_counts = pd.concat(
        [df_arena["model_a"], df_arena["model_b"]]
    ).value_counts()
    well_represented = set(human_battle_counts[human_battle_counts >= 500].index)
    df_arena = df_arena[
        df_arena["model_a"].isin(well_represented)
        & df_arena["model_b"].isin(well_represented)
    ]
    df_results = pd.concat([df_llm_judge, df_arena], ignore_index=True)

    # Bootstrap Bradley-Terry ELO ratings
    n_bootstraps = args.n_bootstraps

    n_llm = len(df_llm_judge)
    n_human = len(df_arena)
    print(f"\n=== ELO Ratings (Bradley-Terry, {n_bootstraps} bootstraps) ===")
    print(
        f"Estimating ELO Ratings with {n_llm} LLM-judges for model {model_name} "
        f"and {n_human} human annotations for other models. Number of battles is indicated in parenthesis and "
        f"confidence intervals are reported by computing ELO on {n_bootstraps} samples of instructions."
    )

    # Count battles per model across the combined results
    battle_counts: dict[str, int] = {}
    for _, row in df_results.iterrows():
        battle_counts[row["model_a"]] = battle_counts.get(row["model_a"], 0) + 1
        battle_counts[row["model_b"]] = battle_counts.get(row["model_b"], 0) + 1

    bootstrap_ratings: list[dict[str, float]] = []
    for _ in range(n_bootstraps):
        df_sample = df_results.sample(
            n=len(df_results), replace=True, random_state=int(rng.integers(0, 2**31))
        )
        ratings = compute_bradley_terry(
            df_sample, winner_col="winner", baseline_model=args.baseline_model
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
    else:
        print("  Not enough data to compute ELO ratings.")

    return {
        **summary,
        "bootstrap_ratings": bootstrap_ratings,
        "model_name": model_name,
    }
