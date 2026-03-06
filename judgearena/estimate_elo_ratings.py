import argparse
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from fast_langdetect import detect_language
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression

from openjury.evaluate import (
    annotate_battles,
    PairScore,
)
from openjury.generate import generate_instructions
from openjury.utils import make_model, cache_function_dataframe


def load_arena_dataframe(
    arena: str, comparia_revision: str = "7a40bce496c1f2aa3be4001da85a49cb4743042b"
) -> pd.DataFrame:
    """
    :param arena:
    :param comparia_revision:
    :return: dataframe containing battles for the arena selected
    """
    assert arena in ["LMArena", "ComparIA"]
    if arena == "LMArena":
        # load LMSys
        path = snapshot_download(
            repo_id="lmarena-ai/arena-human-preference-100k",
            repo_type="dataset",
            allow_patterns="*parquet",
            force_download=False,
        )
        df = pd.read_parquet(
            Path(path) / "data" / "arena-explorer-preference-100k.parquet"
        )
        df["date"] = pd.to_datetime(df["tstamp"], unit="s")
        df["benchmark"] = "LMArena"

    else:
        path = snapshot_download(
            repo_id="ministere-culture/comparia-votes",
            repo_type="dataset",
            allow_patterns="*",
            revision=comparia_revision,
            force_download=False,
        )

        df = pd.read_parquet(Path(path) / "votes.parquet")

        # unify schema
        df["tstamp"] = df["timestamp"]
        df["model_a"] = df["model_a_name"]
        df["model_b"] = df["model_b_name"]

        def get_winner(
            chosen_model_name: str,
            model_a: str,
            model_b: str,
            both_equal: bool,
            **kwargs,
        ):
            if both_equal:
                return "tie"
            else:
                if chosen_model_name is None or isinstance(chosen_model_name, float):
                    return None
                assert chosen_model_name in [
                    model_a,
                    model_b,
                ], f"Chosen model: {chosen_model_name} but model_a: {model_a} and model_b: {model_b}"
                return "model_a" if chosen_model_name == model_a else "model_b"

        df["winner"] = df.apply(lambda row: get_winner(**row), axis=1)
        df = df[~df.winner.isna()]
        df["benchmark"] = "ComparIA"
        df["question_id"] = df["id"]

    cols = [
        "question_id",
        "tstamp",
        "model_a",
        "model_b",
        "winner",
        "conversation_a",
        "conversation_b",
        "benchmark",
    ]
    df = df.loc[:, cols]

    # keep only one turn conversation for now as they are easier to evaluate
    df["turns"] = df.apply(lambda row: len(row["conversation_a"]) - 1, axis=1)
    df = df.loc[df.turns == 1]

    df["lang"] = df.apply(
        lambda row: detect_language(row["conversation_a"][0]["content"]).lower(), axis=1
    )

    return df


@dataclass
class CliEloArgs:
    arena: str
    model: str
    judge_model: str
    n_instructions: int | None = None
    n_instructions_per_language: int | None = None
    languages: list[str] | None = None
    provide_explanation: bool = False
    swap_mode: str = "fixed"
    ignore_cache: bool = False
    truncate_all_input_chars: int = 8192
    max_out_tokens_models: int = 32768
    max_out_tokens_judge: int = 32768
    max_model_len: int | None = None
    chat_template: str | None = None
    result_folder: str = "results"
    engine_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        supported_modes = ["fixed", "both"]
        assert (
            self.swap_mode in supported_modes
        ), f"Only {supported_modes} modes are supported but got {self.swap_mode}."

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(
            prog="Estimate ELO rating for a model on an Arena (LMArena or ComparIA) with LLM judges",
        )
        parser.add_argument(
            "--arena",
            help="The arena to use. Battles are sampled from this Arena.",
            choices=["LMArena", "ComparIA"],
            default="ComparIA",
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
            "--judge",
            "--judge_model",
            dest="judge_model",
            required=True,
            help="Name of the LLM to use, for instance `Together/meta-llama/Meta-Llama-3-70B-Instruct-Turbo`, "
            "`VLLM/meta-llama/Meta-Llama-3-70B-Instruct-Turbo`, `LlamaCpp/path/to/model.gguf` etc",
        )
        parser.add_argument(
            "--n_instructions",
            type=int,
            required=False,
            help="Number of battles used to annotate the model passed with LLM judges.",
        )
        parser.add_argument(
            "--n_instructions_per_language",
            type=int,
            required=False,
            help="Maximum number of instructions to keep per language.",
        )

        parser.add_argument(
            "--provide_explanation",
            action="store_true",
            help="If specified, judge will provide explanation before making a judgement. Does not necessarily improve"
            "the accuracy of the judge but enables some result interpretation.",
        )
        parser.add_argument(
            "--swap_mode",
            type=str,
            choices=["fixed", "both"],
            default="fixed",
            help="Model comparison order mode. 'fixed': always use model order A-B. 'both': correct for model order "
            "bias by evaluating each instruction twice, once as A-B and once as B-A, and average. This helps account "
            "for judge position bias. Default is 'fixed'.",
        )
        parser.add_argument(
            "--ignore_cache",
            action="store_true",
            help="If specified, ignore cache of previous completions.",
        )
        parser.add_argument(
            "--result_folder",
            type=str,
            required=False,
            default="results",
            help="The folder to save the results. Defaults to `results`. Evaluation results will be saved in"
            " `[result_folder]/[evaluation_name]`.",
        )
        parser.add_argument(
            "--truncate_all_input_chars",
            type=int,
            required=False,
            default=8192,
            help="Character-level truncation applied before tokenization: truncates each instruction "
            "before model A/B generation and truncates each completion before judge evaluation.",
        )
        parser.add_argument(
            "--max_out_tokens_models",
            type=int,
            required=False,
            default=32768,
            help=(
                "Generation token budget for each model A/B response. For VLLM, keep this <= "
                "--max_model_len (if provided)."
            ),
        )
        parser.add_argument(
            "--max_out_tokens_judge",
            type=int,
            required=False,
            default=32768,
            help=(
                "Generation token budget for the judge response (reasoning + scores). For "
                "VLLM, keep this <= --max_model_len (if provided)."
            ),
        )
        parser.add_argument(
            "--max_model_len",
            type=int,
            required=False,
            default=None,
            help=(
                "Optional total context window for VLLM models (prompt + generation). This is "
                "independent from --max_out_tokens_models/--max_out_tokens_judge, which only cap "
                "generated tokens. This is useful on smaller GPUs to avoid OOM."
            ),
        )
        parser.add_argument(
            "--chat_template",
            type=str,
            required=False,
            default=None,
            help="Jinja2 chat template string to use instead of the model's tokenizer template. "
            "If not provided, ChatML is used as fallback for models without a chat template.",
        )
        parser.add_argument(
            "--engine_kwargs",
            type=str,
            required=False,
            default="{}",
            help=(
                "JSON dict of engine-specific kwargs forwarded to the underlying engine. "
                'Example for vLLM: \'{"tensor_parallel_size": 2, "gpu_memory_utilization": 0.9}\'.'
            ),
        )
        args = parser.parse_args()

        try:
            engine_kwargs = json.loads(args.engine_kwargs) if args.engine_kwargs else {}
            if not isinstance(engine_kwargs, dict):
                raise ValueError("engine_kwargs must be a JSON object")
        except Exception as e:
            raise SystemExit(f"Failed to parse --engine_kwargs: {e}")

        return cls(
            arena=args.arena,
            model=args.model,
            judge_model=args.judge_model,
            n_instructions=args.n_instructions,
            n_instructions_per_language=args.n_instructions_per_language,
            languages=args.languages,
            provide_explanation=args.provide_explanation,
            swap_mode=args.swap_mode,
            ignore_cache=args.ignore_cache,
            truncate_all_input_chars=args.truncate_all_input_chars,
            max_out_tokens_models=args.max_out_tokens_models,
            max_out_tokens_judge=args.max_out_tokens_judge,
            max_model_len=args.max_model_len,
            chat_template=args.chat_template,
            result_folder=args.result_folder,
            engine_kwargs=engine_kwargs,
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

    lr = LogisticRegression(fit_intercept=False, C=1e10, tol=1e-6, max_iter=500)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = scale * lr.coef_[0] + init_rating

    # Normalize to baseline model if specified
    if baseline_model is not None and baseline_model in models.index:
        elo_scores += baseline_rating - elo_scores[models[baseline_model]]

    return dict(pd.Series(elo_scores, index=models.index))


def main():
    args = CliEloArgs.parse_args()

    seed = 0
    rng = np.random.default_rng(seed)

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
        [row["conversation_a"][0]["content"] for _, row in df_battles.iterrows()],
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

    cache_suffix = (
        f"{args.arena}_{args.model.replace("/", "_")}_"
        f"{args.judge_model.replace("/", "_")}_{args.n_instructions}_{args.n_instructions_per_language}"
    )
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
            row["conversation_a"][1]["content"]
            if use_model_a_as_opponent[i]
            else row["conversation_b"][1]["content"]
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
        anns = annotate_battles(
            judge_chat_model=judge_chat_model,
            instructions=instructions.tolist(),
            completions_A=completions_A,
            completions_B=completions_B,
            provide_explanation=args.provide_explanation,
            truncate_input_chars=args.truncate_all_input_chars,
            use_tqdm=use_tqdm,
        )
        return pd.DataFrame(
            {
                "judge_completion": [a.judge_completion for a in anns],
                "instruction": [a.instruction for a in anns],
                "completion_A": [a.completion_A for a in anns],
                "completion_B": [a.completion_B for a in anns],
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

    # Restore position arrays from cache (in case loaded from disk)
    use_model_a_as_opponent = df_judge["use_model_a_as_opponent"].to_numpy()
    our_model_is_position_a = df_judge["our_model_is_position_a"].to_numpy()
    opponent_models = df_judge["opponent_model"].tolist()

    print(f"First judge output:\n{df_judge['judge_completion'].iloc[0][:500]}\n")

    # Parse preferences: <0.5 means A wins, >0.5 means B wins, None means unparseable
    score_parser = PairScore()
    prefs = [score_parser.parse_model_raw(jc) for jc in df_judge["judge_completion"]]

    # Map preferences back to model-name-level battle results
    model_name = args.model
    battle_results = []
    for i, (pref, is_pos_a, opp_model) in enumerate(
        zip(prefs, our_model_is_position_a, opponent_models)
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

    # Compute win/loss/tie counts and winrate
    our_wins = sum(
        1
        for r in battle_results
        if (r["model_a"] == model_name and r["winner"] == "model_a")
        or (r["model_b"] == model_name and r["winner"] == "model_b")
    )
    our_losses = sum(
        1
        for r in battle_results
        if (r["model_a"] == model_name and r["winner"] == "model_b")
        or (r["model_b"] == model_name and r["winner"] == "model_a")
    )
    our_ties = sum(1 for r in battle_results if r["winner"] == "tie")
    winrate = (our_wins + 0.5 * our_ties) / n if n > 0 else 0.0

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
    n_bootstraps = 10

    n_llm = len(df_llm_judge)
    n_human = len(df_arena)
    print(f"\n=== ELO Ratings (Bradley-Terry, 10 bootstraps) ===")
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
        try:
            ratings = compute_bradley_terry(df_sample, winner_col="winner")
            bootstrap_ratings.append(ratings)
        except Exception:
            pass

    if bootstrap_ratings:
        all_model_names = sorted(
            set(df_results["model_a"]) | set(df_results["model_b"])
        )
        mean_ratings = {
            m: np.mean([r.get(m, 1000) for r in bootstrap_ratings])
            for m in all_model_names
        }
        for m in sorted(all_model_names, key=lambda x: -mean_ratings[x]):
            vals = [r.get(m, 1000) for r in bootstrap_ratings]
            suffix = " <-----" if m == model_name else ""
            count = battle_counts.get(m, 0)
            print(f"  {m}  ({count}){suffix}: {np.mean(vals):.1f} ± {np.std(vals):.1f}")
    else:
        print("  Not enough data to compute ELO ratings.")


if __name__ == "__main__":
    main()
