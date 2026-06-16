"""Preference statistics and human-readable result reporting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

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


class Report(ABC):
    """A reportable result that can print itself and serialize to a dict."""

    @abstractmethod
    def render(self) -> None: ...

    @abstractmethod
    def to_dict(self) -> dict: ...


@dataclass
class BattleReport(Report):
    """Pairwise battle results for the arena and MT-Bench pipelines."""

    task: str
    model_a: str
    model_b: str
    judge_model: str
    summary: PrefSummary
    swap_mode: str | None = None
    result_folder: str | None = None
    per_category: dict | None = None
    per_turn: dict | None = None
    preferences: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict = {
            "task": self.task,
            "model_A": self.model_a,
            "model_B": self.model_b,
            "judge_model": self.judge_model,
            **self.summary.to_dict(),
        }
        if self.swap_mode is not None:
            out["swap_mode"] = self.swap_mode
        if self.result_folder is not None:
            out["result_folder"] = self.result_folder
        if self.per_category is not None:
            out["per_category"] = self.per_category
        if self.per_turn is not None:
            out["per_turn"] = self.per_turn
        out.update(self.metadata)
        out["preferences"] = self.preferences
        return out

    def render(self) -> None:
        s = self.summary
        print("\n" + "=" * 60)
        print("🏆 MODEL BATTLE RESULTS 🏆".center(60))
        print(f"📊 Task: {self.task}")
        print(f"🤖 Competitors: Model A: {self.model_a} vs Model B: {self.model_b}")
        print(f"⚖️ Judge: {self.judge_model}")
        print("📈 Results Summary:")
        if s.num_missing > 0:
            parsed = s.num_battles - s.num_missing
            print(
                f"   Total Battles: {s.num_battles}  ⚠️  {s.num_missing} unparseable "
                f"(parsed: {parsed}/{s.num_battles})"
            )
        elif self.swap_mode == "both":
            print(
                f"   Total Battles: {s.num_battles} (2×{s.num_battles // 2} — each "
                f"instruction judged in both orders to detect positional bias)"
            )
        else:
            print(f"   Total Battles: {s.num_battles}")
        print(f"   Win Rate (A): {s.winrate:.1%}")
        print(f"   ✅ Wins:   {s.num_wins}")
        print(f"   ❌ Losses: {s.num_losses}")
        print(f"   🤝 Ties:   {s.num_ties}")

        if self.per_category:
            print("\nPer-Category Breakdown:")
            print(
                f"  {'Category':<14} | {'Win Rate(A)':>11} | "
                f"{'Wins':>4} | {'Losses':>6} | {'Ties':>4}"
            )
            print(f"  {'-' * 14}-+-{'-' * 11}-+-{'-' * 4}-+-{'-' * 6}-+-{'-' * 4}")
            for cat, stats in sorted(self.per_category.items()):
                print(
                    f"  {cat:<14} | {stats['winrate']:>11.1%} | "
                    f"{stats['num_wins']:>4} | {stats['num_losses']:>6} | "
                    f"{stats['num_ties']:>4}"
                )

        if self.per_turn:
            print("\nPer-Turn Breakdown:")
            for turn, stats in sorted(self.per_turn.items()):
                print(
                    f"  Turn {turn} Win Rate(A): {stats['winrate']:.1%} "
                    f"(W:{stats['num_wins']} L:{stats['num_losses']} T:{stats['num_ties']})"
                )

        if self.result_folder:
            print(f"📁 Results: {self.result_folder}")
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
