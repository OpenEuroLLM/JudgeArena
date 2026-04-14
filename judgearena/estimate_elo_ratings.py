import argparse
import hashlib
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from judgearena.arenas_utils import _extract_instruction_text, load_arena_dataframe
from judgearena.cli_common import BaseCliArgs, add_common_arguments, parse_engine_kwargs
from judgearena.evaluate import judge_and_parse_prefs
from judgearena.generate import generate_instructions
from judgearena.utils import cache_function_dataframe, compute_pref_summary, make_model


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

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(
            prog="Estimate ELO rating for a model on an Arena (LMArena-100k, LMArena-140k, or ComparIA) with LLM judges",
        )
        parser.add_argument(
            "--arena",
            help="The arena to use. Battles are sampled from this Arena. If not passed use concatenation from all Arena. "
            "Passing LMArena leads to loading the union of `LMArena-100k` and `LMArena-140k`",
            choices=["LMArena-100k", "LMArena-140k", "ComparIA", "LMArena"],
            required=False,
        )
        parser.add_argument(
            "--model",
            required=True,
            help="Name of the LLM to use for a generation, must be a valid choice for `generation_provider`",
        )
        parser.add_argument(
            "--languages",
            nargs="+",
            default=None,
            help='List of language codes to evaluate, e.g. "en fr de" (default: all languages)',
        )
        parser.add_argument(
            "--n_instructions_per_language",
            type=int,
            required=False,
            help="Maximum number of instructions to keep per language.",
        )
        parser.add_argument(
            "--n_bootstraps",
            type=int,
            required=False,
            default=20,
            help="Number of bootstrap samples for ELO confidence intervals. Default is 20.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            required=False,
            default=0,
            help="Random seed for reproducibility. Default is 0.",
        )
        parser.add_argument(
            "--baseline_model",
            type=str,
            required=False,
            default=None,
            help="Model name to anchor at 1000 ELO. All other ratings are expressed relative to this model. "
            "Must be one of the models present in the arena battles. If not set, ratings are not anchored.",
        )
        parser.add_argument(
            "--soft-elo",
            action="store_true",
            help="Use continuous judge preferences as soft labels for BT fitting "
            "instead of discretising to hard win/loss/tie.",
        )
        add_common_arguments(parser)
        args = parser.parse_args()

        return cls(
            arena=args.arena,
            model=args.model,
            n_instructions_per_language=args.n_instructions_per_language,
            languages=args.languages,
            n_bootstraps=args.n_bootstraps,
            seed=args.seed,
            baseline_model=args.baseline_model,
            soft_elo=args.soft_elo,
            judge_model=args.judge_model,
            n_instructions=args.n_instructions,
            provide_explanation=args.provide_explanation,
            swap_mode=args.swap_mode,
            ignore_cache=args.ignore_cache,
            truncate_all_input_chars=args.truncate_all_input_chars,
            max_out_tokens_models=args.max_out_tokens_models,
            max_out_tokens_judge=args.max_out_tokens_judge,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            result_folder=args.result_folder,
            engine_kwargs=parse_engine_kwargs(args.engine_kwargs),
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


def compute_soft_bradley_terry(
    df: pd.DataFrame,
    pref_col: str = "pref",
    scale: float = 400,
    base: float = 10,
    init_rating: float = 1000,
    baseline_model: str | None = None,
    baseline_rating: float = 1000,
) -> dict[str, float]:
    """Compute Bradley-Terry ratings from continuous (soft) preferences.

    Each row in *df* is a single battle with columns ``model_a``, ``model_b``,
    and *pref_col* ∈ [0, 1] where 0 → A wins, 1 → B wins, 0.5 → tie.

    The soft cross-entropy for a single battle is decomposed into two
    weighted hard-label rows so that sklearn ``LogisticRegression`` can be
    reused:

        row 1: Y=1, weight = 1 - pref   (evidence for A winning)
        row 2: Y=0, weight = pref        (evidence for B winning)
    """
    df = df.dropna(subset=[pref_col]).copy()
    if df.empty:
        return {}

    all_models = sorted(set(df["model_a"].unique()) | set(df["model_b"].unique()))
    models = pd.Series(np.arange(len(all_models)), index=all_models)
    p = len(models)

    n_battles = len(df)
    X = np.zeros([2 * n_battles, p])
    Y = np.zeros(2 * n_battles)
    sample_weights = np.zeros(2 * n_battles)

    for idx, (_, row) in enumerate(df.iterrows()):
        m_a = row["model_a"]
        m_b = row["model_b"]
        pref = row[pref_col]

        # Row for "A wins" evidence
        X[2 * idx, models[m_a]] = +np.log(base)
        X[2 * idx, models[m_b]] = -np.log(base)
        Y[2 * idx] = 1.0
        sample_weights[2 * idx] = 1.0 - pref

        # Row for "B wins" evidence
        X[2 * idx + 1, models[m_a]] = +np.log(base)
        X[2 * idx + 1, models[m_b]] = -np.log(base)
        Y[2 * idx + 1] = 0.0
        sample_weights[2 * idx + 1] = pref

    # Drop rows with zero weight (pure wins have one side = 0)
    nonzero = sample_weights > 0
    X = X[nonzero]
    Y = Y[nonzero]
    sample_weights = sample_weights[nonzero]

    if len(X) == 0:
        return {}

    lr = LogisticRegression(fit_intercept=False, C=1e10, tol=1e-6, max_iter=1000)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = scale * lr.coef_[0] + init_rating

    if baseline_model is not None and baseline_model in models.index:
        elo_scores += baseline_rating - elo_scores[models[baseline_model]]

    return dict(pd.Series(elo_scores, index=models.index))


def _winner_to_pref(winner: str) -> float | None:
    """Convert a hard winner label to a continuous preference value."""
    if winner == "model_a":
        return 0.0
    elif winner == "model_b":
        return 1.0
    elif winner in ("tie", "tie (bothbad)"):
        return 0.5
    return None


def main(args: CliEloArgs | None = None) -> dict:
    if args is None:
        args = CliEloArgs.parse_args()

    rng = np.random.default_rng(args.seed)

    # Step 1: Load arena battles
    print(f"\n=== Step 1: Loading battles from {args.arena} ===")
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
    print(f"Loaded {n} battles.")

    # Extract user instructions (first turn of conversation_a)
    instructions = pd.Series(
        [
            _extract_instruction_text(row["conversation_a"][0])
            for _, row in df_battles.iterrows()
        ],
        name="instruction",
    )
    print(f"\nFirst instruction:\n{instructions.iloc[0][:300]}\n")

    # Step 2: Generate completions for the model under evaluation
    print(f"=== Step 2: Generating completions with {args.model} ===")

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
        print(
            f"Cache suffix too long ({len(cache_suffix)} chars), using hash: {cache_hash} (full: {cache_suffix})"
        )
        cache_suffix = cache_hash
    completions_df = cache_function_dataframe(
        lambda: gen_fun(instructions=instructions, model=args.model),
        ignore_cache=args.ignore_cache,
        cache_name=f"elo/{cache_suffix}",
    ).set_index("instruction_index")
    completions = completions_df.loc[:, "completion"]

    print(f"First completion:\n{completions.iloc[0]}\n")

    # Step 3: Judge evaluation against randomly picked arena opponents
    print(f"=== Step 3: Judge evaluation with {args.judge_model} ===")

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
        annotations, _, prefs = judge_and_parse_prefs(
            judge_chat_model=judge_chat_model,
            instructions=instructions.tolist(),
            completions_A=completions_A,
            completions_B=completions_B,
            swap_mode=args.swap_mode,
            provide_explanation=args.provide_explanation,
            truncate_input_chars=args.truncate_all_input_chars,
            use_tqdm=use_tqdm,
        )
        return pd.DataFrame(
            {
                "judge_completion": [a.judge_completion for a in annotations],
                "instruction": [a.instruction for a in annotations],
                "completion_A": [a.completion_A for a in annotations],
                "completion_B": [a.completion_B for a in annotations],
                "pref": prefs,
                "use_model_a_as_opponent": use_model_a_as_opponent,
                "our_model_is_position_a": our_model_is_position_a,
                "opponent_model": opponent_models,
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

    print(f"First judge output:\n{df_judge['judge_completion'].iloc[0][:500]}\n")

    # Map preferences back to model-name-level battle results.
    # Build both hard labels (winner) and continuous prefs for each battle.
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

        # Continuous pref is relative to judge positions (A/B).
        # Remap so that model_a column in the DataFrame always corresponds
        # to pref=0 and model_b to pref=1.
        if is_pos_a:
            battle_results.append(
                {"model_a": model_name, "model_b": opp_model, "winner": winner, "pref": pref}
            )
        else:
            battle_results.append(
                {"model_a": opp_model, "model_b": model_name, "winner": winner, "pref": 1.0 - pref if pref is not None else None}
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
    df_arena = df_arena_all.loc[:, ["model_a", "model_b", "winner"]].copy()
    human_battle_counts = pd.concat(
        [df_arena["model_a"], df_arena["model_b"]]
    ).value_counts()
    well_represented = set(human_battle_counts[human_battle_counts >= 500].index)
    df_arena = df_arena[
        df_arena["model_a"].isin(well_represented)
        & df_arena["model_b"].isin(well_represented)
    ]
    # Add pref column to arena battles (hard labels → 0.0 / 1.0 / 0.5)
    df_arena["pref"] = df_arena["winner"].map(_winner_to_pref)

    df_results = pd.concat([df_llm_judge, df_arena], ignore_index=True)

    # Compute human-only BT ratings as ground-truth reference
    human_elo = compute_bradley_terry(
        df_arena, winner_col="winner", baseline_model=args.baseline_model
    )

    # Bootstrap Bradley-Terry ELO ratings
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

    bootstrap_ratings: list[dict[str, float]] = []
    for _ in range(n_bootstraps):
        df_sample = df_results.sample(
            n=len(df_results), replace=True, random_state=int(rng.integers(0, 2**31))
        )
        if use_soft:
            ratings = compute_soft_bradley_terry(
                df_sample, pref_col="pref", baseline_model=args.baseline_model
            )
        else:
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

    return {
        **summary,
        "bootstrap_ratings": bootstrap_ratings,
        "human_elo": human_elo,
        "mae_vs_human": mae,
        "model_name": model_name,
        "method": method_label,
    }


def cli():
    main()


if __name__ == "__main__":
    cli()
