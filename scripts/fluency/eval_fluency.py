"""
Evaluate LLM judge accuracy on the ltg/normistral-fluency-annotation dataset.

Downloads the dataset, runs an LLM judge on each pair, and computes accuracy
against human annotations.

Usage:
    python scripts/fluency/eval_fluency.py --judge_model OpenRouter/google/gemma-4-31b-it
"""

import argparse

import pandas as pd
from datasets import load_dataset
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from judgearena.evaluate import PairScore, annotate_battles
from judgearena.models import make_model
from judgearena.prompts.registry import FLUENCY_JUDGE_PROMPT_PRESET
from judgearena.utils import data_root


def set_langchain_cache():
    set_llm_cache(SQLiteCache(database_path=str(data_root / ".langchain.db")))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM judge on Norwegian fluency annotations."
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="OpenRouter/google/gemma-4-31b-it",
        help="Judge model in openjury format, e.g. 'OpenRouter/openai/gpt-4o-mini'",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Max tokens for judge output.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (default: all).",
    )
    parser.add_argument(
        "--use_tqdm",
        action="store_true",
        help="Show progress bar during inference.",
    )
    parser.add_argument(
        "--both",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run judge twice (original and swapped A/B order) and average preferences to correct for position bias (default: True).",
    )
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        help=(
            "Skip the LangChain SQLite completion cache. This dataset has many exact-"
            "duplicate (prompt, response_a, response_b) rows (e.g. multiple annotators "
            "judging the same pair), so concurrent identical judge calls can race on a "
            "SQLite UNIQUE constraint when writing to the cache. Pass this flag to avoid "
            "that crash (at the cost of no caching for this run)."
        ),
    )
    return parser.parse_args()


def human_choice_to_label(choice: str) -> str | None:
    """Map human annotation choice string to 'A', 'B', or 'tie'."""
    choice_lower = choice.lower().strip()
    if "a" in choice_lower and "more fluent" in choice_lower:
        return "A"
    elif "b" in choice_lower and "more fluent" in choice_lower:
        return "B"
    elif "tie" in choice_lower or "both" in choice_lower or "equal" in choice_lower:
        return "tie"
    return None


def preference_to_label(pref: float | None) -> str:
    """Convert a preference score to 'A', 'B', or 'tie'."""
    if pref is None:
        return "unknown"
    if pref < 0.5:
        return "A"
    elif pref > 0.5:
        return "B"
    else:
        return "tie"


def main():
    args = parse_args()
    if not args.ignore_cache:
        set_langchain_cache()

    print("Loading dataset ltg/normistral-fluency-annotation...")
    ds = load_dataset("ltg/normistral-fluency-annotation", split="test")
    df = ds.to_pandas()
    print(f"Loaded {len(df)} rows.")

    if args.n_samples is not None:
        df = df.head(args.n_samples)
        print(f"Using first {len(df)} samples.")

    # The dataset has many exact-duplicate (prompt, response_a, response_b) rows
    # -- e.g. the same pair judged by multiple annotators -- so judge only the
    # unique comparisons and broadcast the verdict back to every row. This
    # roughly halves the number of judge calls and avoids submitting identical
    # prompts concurrently (which used to race on the SQLite completion cache).
    dedup_cols = ["prompt", "response_a", "response_b"]
    df_unique = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    print(
        f"{len(df_unique)} unique (prompt, response_a, response_b) comparisons "
        f"out of {len(df)} rows."
    )

    instructions = df_unique["prompt"].tolist()
    completions_A = df_unique["response_a"].tolist()
    completions_B = df_unique["response_b"].tolist()

    print(f"Running LLM judge: {args.judge_model}")
    judge_model = make_model(args.judge_model, max_tokens=args.max_tokens)

    annotations = annotate_battles(
        judge_chat_model=judge_model,
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        prompt_preset=FLUENCY_JUDGE_PROMPT_PRESET,
        use_tqdm=args.use_tqdm,
    )

    score_parser = PairScore()
    prefs = [score_parser.parse_model_raw(ann.judge_completion) for ann in annotations]

    if args.both:
        print("Running judge again with A and B swapped...")
        annotations_swapped = annotate_battles(
            judge_chat_model=judge_model,
            instructions=instructions,
            completions_A=completions_B,
            completions_B=completions_A,
            prompt_preset=FLUENCY_JUDGE_PROMPT_PRESET,
            use_tqdm=args.use_tqdm,
        )
        prefs_swapped = [
            score_parser.parse_model_raw(ann.judge_completion)
            for ann in annotations_swapped
        ]
        # Reconstruct preference for original A: average pref and (1 - pref_swapped)
        prefs = [
            (
                (p + (1 - q)) / 2
                if p is not None and q is not None
                else (p if p is not None else (1 - q) if q is not None else None)
            )
            for p, q in zip(prefs, prefs_swapped, strict=True)
        ]

    # Attach judge verdicts to the unique comparisons, then broadcast them back
    # onto every (possibly duplicated) row of the original dataset.
    df_unique = df_unique.copy()
    df_unique["judge_preference"] = prefs
    df_unique["judge_completion"] = [ann.judge_completion for ann in annotations]

    df = df.merge(
        df_unique[[*dedup_cols, "judge_preference", "judge_completion"]],
        on=dedup_cols,
        how="left",
    )
    df["judge_label"] = [preference_to_label(p) for p in df["judge_preference"]]
    df["human_label"] = [human_choice_to_label(c) for c in df["choice"]]

    # Restrict to non-tie comparisons with parseable judge output. Following
    # the methodology of the paper that introduced this dataset, ties are
    # excluded from accuracy reporting.
    agreed = df[
        df["human_label"].notna()
        & (df["human_label"] != "tie")
        & (df["judge_label"] != "unknown")
    ]
    n_total = len(df)
    n_agreed = len(agreed)

    print(f"\nTotal samples: {n_total}")
    print(f"Non-tie comparisons with parseable judge output: {n_agreed}")

    # Accuracy excluding ties
    correct_agreed = (agreed["judge_label"] == agreed["human_label"]).sum()
    accuracy_agreed = correct_agreed / n_agreed if n_agreed > 0 else float("nan")
    print(
        f"\nAccuracy LLM judge (excluding ties): {correct_agreed}/{n_agreed} = {accuracy_agreed:.3f}"
    )

    if "annotator_id" in df.columns:
        annotator_accs = []
        print("\nAccuracy per annotator (excluding ties):")
        for annotator, group in agreed.groupby("annotator_id"):
            acc = (group["judge_label"] == group["human_label"]).sum() / len(group)
            annotator_accs.append(acc)
            print(
                f"  {annotator}: {(group['judge_label'] == group['human_label']).sum()}/{len(group)} = {acc:.3f}"
            )
        if annotator_accs:
            print(
                f"  Average annotator accuracy: {sum(annotator_accs) / len(annotator_accs):.3f}"
            )

    # Distribution of human labels (excluding ties)
    print("\nHuman label distribution (excluding ties):")
    print(agreed["human_label"].value_counts().to_string())

    # Distribution of judge labels (excluding ties)
    print("\nJudge label distribution (excluding ties):")
    print(agreed["judge_label"].value_counts().to_string())

    # Accuracy per human label (excluding ties)
    print("\nAccuracy per human label:")
    for label in ["A", "B"]:
        subset = agreed[agreed["human_label"] == label]
        if len(subset) == 0:
            continue
        acc = (subset["judge_label"] == label).sum() / len(subset)
        print(
            f"  {label}: {(subset['judge_label'] == label).sum()}/{len(subset)} = {acc:.3f}"
        )

    # Confusion matrix (excluding ties)
    print("\nConfusion matrix (rows=human, cols=judge, excluding ties):")
    conf = pd.crosstab(
        agreed["human_label"],
        agreed["judge_label"],
        rownames=["human"],
        colnames=["judge"],
    )
    print(conf.to_string())


if __name__ == "__main__":
    main()
