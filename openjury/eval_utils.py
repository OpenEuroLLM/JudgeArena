"""Shared evaluation runtime helpers used by entrypoints and benchmark pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from openjury.evaluate import annotate_battles, PairScore
from openjury.utils import compute_pref_summary


def print_results(results):
    """Print battle results in a readable format."""
    print("\n" + "=" * 60)
    print("🏆 MODEL BATTLE RESULTS 🏆".center(60))
    print(f"📊 Dataset: {results['dataset']}")
    print(
        f"🤖 Competitors: Model A: {results['model_A']} vs Model B: {results['model_B']}"
    )
    print(f"⚖️ Judge: {results['judge_model']}")
    print("📈 Results Summary:")
    print(f"   Total Battles: {results['num_battles']}")
    print(f"   Win Rate (A): {results['winrate']:.1%}")
    print(f"   ✅ Wins:   {results['num_wins']}")
    print(f"   ❌ Losses: {results['num_losses']}")
    print(f"   🤝 Ties:   {results['num_ties']}")
    if results.get("num_missing", 0) > 0:
        print(f"   ❓ Missing: {results['num_missing']}")

    per_category = results.get("per_category")
    if per_category:
        print("\nPer-Category Breakdown:")
        print(
            f"  {'Category':<14} | {'Win Rate(A)':>11} | {'Wins':>4} | {'Losses':>6} | {'Ties':>4}"
        )
        print(f"  {'-' * 14}-+-{'-' * 11}-+-{'-' * 4}-+-{'-' * 6}-+-{'-' * 4}")
        for cat, stats in sorted(per_category.items()):
            print(
                f"  {cat:<14} | {stats['winrate']:>11.1%} | "
                f"{stats['num_wins']:>4} | {stats['num_losses']:>6} | {stats['num_ties']:>4}"
            )

    per_turn = results.get("per_turn")
    if per_turn:
        print("\nPer-Turn Breakdown:")
        for turn, stats in sorted(per_turn.items()):
            print(
                f"  Turn {turn} Win Rate(A): {stats['winrate']:.1%} "
                f"(W:{stats['num_wins']} L:{stats['num_losses']} T:{stats['num_ties']})"
            )
    print("=" * 60 + "\n")


def _compute_grouped_stats(
    preferences: pd.Series,
    metadata: list[dict[str, object]],
    group_by: str,
) -> dict[object, dict[str, float | int]]:
    grouped: dict[object, list[float]] = {}
    for meta, pref in zip(metadata, preferences):
        key = meta.get(group_by)
        if key is None:
            continue
        grouped.setdefault(key, []).append(pref)
    return {
        key: compute_pref_summary(pd.Series(vals))
        for key, vals in grouped.items()
    }


def _parse_preferences_from_annotations(
    annotations: list,
    score_parser: PairScore,
) -> pd.Series:
    return pd.Series(
        [
            score_parser.parse_model_raw(annotation.judge_completion)
            for annotation in annotations
        ]
    )


@dataclass
class JudgeAnnotationResult:
    annotations: list
    annotations_reversed: list
    metadata_for_annotations: list[dict[str, object]]
    metadata_for_reversed_annotations: list[dict[str, object]]
    preferences: pd.Series
    combined_metadata: list[dict[str, object]]


def _make_judge_annotation(
    *,
    judge_chat_model,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    metadata: list[dict[str, object]],
    score_parser: PairScore,
    provide_explanation: bool,
    swap_mode: str,
    truncate_input_chars: int | None,
    use_tqdm: bool,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> JudgeAnnotationResult:
    if not instructions:
        raise ValueError("instructions must be non-empty")

    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        provide_explanation=provide_explanation,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        truncate_input_chars=truncate_input_chars,
        use_tqdm=use_tqdm,
    )
    preference_parts = [_parse_preferences_from_annotations(annotations, score_parser)]

    annotations_reversed: list = []
    metadata_for_reversed_annotations: list[dict[str, object]] = []
    combined_metadata = list(metadata)

    if swap_mode == "both":
        print("Correction for judge bias towards a certain model position is set.")
        print("Evaluating completions with models reversed.")
        annotations_reversed = annotate_battles(
            judge_chat_model=judge_chat_model,
            instructions=instructions,
            completions_A=completions_B,
            completions_B=completions_A,
            provide_explanation=provide_explanation,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            truncate_input_chars=truncate_input_chars,
            use_tqdm=use_tqdm,
        )
        prefs_reversed = _parse_preferences_from_annotations(
            annotations_reversed, score_parser
        )
        preference_parts.append(1 - prefs_reversed)
        metadata_for_reversed_annotations = list(metadata)
        combined_metadata.extend(metadata)

    preferences = pd.concat(preference_parts).reset_index(drop=True)
    return JudgeAnnotationResult(
        annotations=annotations,
        annotations_reversed=annotations_reversed,
        metadata_for_annotations=list(metadata),
        metadata_for_reversed_annotations=metadata_for_reversed_annotations,
        preferences=preferences,
        combined_metadata=combined_metadata,
    )
