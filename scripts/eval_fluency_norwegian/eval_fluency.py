"""
Evaluate LLM judge accuracy on the ltg/normistral-fluency-annotation dataset.

Downloads the dataset, runs an LLM judge on each pair, and computes accuracy
against human annotations.

Usage:
    python scripts/eval_fluency_norwegian/eval_fluency.py --judge_model OpenRouter/openai/gpt-4o-mini
    python scripts/eval_fluency_norwegian/eval_fluency.py --judge_model Together/meta-llama/Llama-3.3-70B-Instruct-Turbo
"""

import argparse

import pandas as pd
from datasets import load_dataset

from judgearena.evaluate import PairScore, annotate_battles
from judgearena.utils import make_model, set_langchain_cache

FLUENCY_SYSTEM_PROMPT = (
    "You are a highly efficient assistant, who evaluates and selects the best large language "
    "model based on the quality of completion of a sentence. You will see a sentence to be completed and two "
    "completions from Assistant A and Assistant B and will have to decide which one is best. Make sure to not "
    "over-confidently prefer one assistant or the other and also make sure to not bias your preference based on "
    "the ordering or on the length of the answers."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM judge on Norwegian fluency annotations."
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="OpenRouter/deepseek/deepseek-v3.2",
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
    return parser.parse_args()


def choice_to_label(choice: str) -> str | None:
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
    set_langchain_cache()

    print("Loading dataset ltg/normistral-fluency-annotation...")
    ds = load_dataset("ltg/normistral-fluency-annotation", split="test")
    df = ds.to_pandas()
    print(f"Loaded {len(df)} rows.")

    if args.n_samples is not None:
        df = df.head(args.n_samples)
        print(f"Using first {len(df)} samples.")

    instructions = df["prompt"].tolist()
    completions_A = df["response_a"].tolist()
    completions_B = df["response_b"].tolist()

    print(f"Running LLM judge: {args.judge_model}")
    judge_model = make_model(args.judge_model, max_tokens=args.max_tokens)

    annotations = annotate_battles(
        judge_chat_model=judge_model,
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        system_prompt=FLUENCY_SYSTEM_PROMPT,
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
            system_prompt=FLUENCY_SYSTEM_PROMPT,
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

    # Build results dataframe
    df = df.copy()
    df["judge_preference"] = prefs
    df["judge_label"] = [preference_to_label(p) for p in prefs]
    df["human_label"] = [choice_to_label(c) for c in df["choice"]]
    df["judge_completion"] = [ann.judge_completion for ann in annotations]

    # Filter to rows where both labels are valid (non-null, non-unknown)
    valid = df[df["human_label"].notna() & (df["judge_label"] != "unknown")]
    n_total = len(df)
    n_valid = len(valid)
    n_parse_fail = n_total - n_valid

    print(f"\nTotal samples: {n_total}")
    print(f"Parse failures (judge output not parseable): {n_parse_fail}")
    print(f"Valid comparisons: {n_valid}")

    # Overall accuracy
    correct = (valid["judge_label"] == valid["human_label"]).sum()
    accuracy = correct / n_valid if n_valid > 0 else float("nan")
    print(f"\nOverall accuracy: {correct}/{n_valid} = {accuracy:.3f}")

    # Accuracy on agreed pairs only (human label is A or B, not tie)
    agreed = valid[valid["human_label"] != "tie"]
    n_agreed = len(agreed)
    correct_agreed = (agreed["judge_label"] == agreed["human_label"]).sum()
    accuracy_agreed = correct_agreed / n_agreed if n_agreed > 0 else float("nan")
    print(
        f"Accuracy (annotator preferred one response, excluding ties): "
        f"{correct_agreed}/{n_agreed} = {accuracy_agreed:.3f}"
    )
    if "annotator_id" in df.columns:
        annotator_accs = []
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

    # Distribution of human labels
    print("\nHuman label distribution:")
    print(valid["human_label"].value_counts().to_string())

    # Distribution of judge labels
    print("\nJudge label distribution:")
    print(valid["judge_label"].value_counts().to_string())

    # Accuracy per human label
    print("\nAccuracy per human label:")
    for label in ["A", "B", "tie"]:
        subset = valid[valid["human_label"] == label]
        if len(subset) == 0:
            continue
        acc = (subset["judge_label"] == label).sum() / len(subset)
        print(
            f"  {label}: {(subset['judge_label'] == label).sum()}/{len(subset)} = {acc:.3f}"
        )

    # Accuracy per annotator
    if "annotator_id" in df.columns:
        print("\nAccuracy per annotator:")
        for annotator, group in valid.groupby("annotator_id"):
            acc = (group["judge_label"] == group["human_label"]).sum() / len(group)
            print(
                f"  {annotator}: {(group['judge_label'] == group['human_label']).sum()}/{len(group)} = {acc:.3f}"
            )

    # Confusion matrix
    print("\nConfusion matrix (rows=human, cols=judge):")
    conf = pd.crosstab(
        valid["human_label"],
        valid["judge_label"],
        rownames=["human"],
        colnames=["judge"],
    )
    print(conf.to_string())


if __name__ == "__main__":
    main()
