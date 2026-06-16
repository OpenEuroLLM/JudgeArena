"""Preference statistics and human-readable result reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class PrefSummary:
    """Win/loss/tie statistics for a preference series (0=A, 0.5=tie, 1=B)."""

    num_battles: int
    winrate: float
    num_wins: int
    num_losses: int
    num_ties: int
    num_missing: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_pref_summary(prefs: pd.Series) -> PrefSummary:
    """Compute win/loss/tie stats for preference series (0=A, 0.5=tie, 1=B)."""
    prefs = pd.Series(prefs, dtype="float64")
    valid = prefs.dropna()
    num_wins = int((valid < 0.5).sum())
    num_losses = int((valid > 0.5).sum())
    num_ties = int((valid == 0.5).sum())
    num_battles = int(len(prefs))
    denom = num_wins + num_losses + num_ties
    winrate = float((num_wins + 0.5 * num_ties) / denom) if denom > 0 else float("nan")
    return PrefSummary(
        num_battles=num_battles,
        winrate=winrate,
        num_wins=num_wins,
        num_losses=num_losses,
        num_ties=num_ties,
        num_missing=int(num_battles - denom),
    )


def print_results(results):
    """Print battle results in a readable format."""
    print("\n" + "=" * 60)
    print("🏆 MODEL BATTLE RESULTS 🏆".center(60))
    print(f"📊 Task: {results['task']}")
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
    for meta, pref in zip(metadata, preferences, strict=True):
        key = meta.get(group_by)
        if key is None:
            continue
        grouped.setdefault(key, []).append(pref)
    return {key: compute_pref_summary(pd.Series(vals)).to_dict() for key, vals in grouped.items()}
