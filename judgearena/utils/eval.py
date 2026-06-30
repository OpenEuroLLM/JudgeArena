"""Preference statistics and human-readable result reporting."""

from __future__ import annotations

import abc
import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_serializer


class PrefSummary(BaseModel):
    """Win/loss/tie statistics for a preference series (0=A, 0.5=tie, 1=B)."""

    num_battles: int
    winrate: float
    num_wins: int
    num_losses: int
    num_ties: int
    num_missing: int

    def to_dict(self) -> dict[str, float | int]:
        return self.model_dump()


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


class Report(BaseModel, abc.ABC):
    """A reportable result that renders, serializes (versioned), and saves itself."""

    # protected_namespaces=() allows model_* field names (model_a, model_name)
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        protected_namespaces=(),
        use_attribute_docstrings=True,
    )

    @computed_field
    @property
    def schema_version(self) -> str:
        return "1"

    @computed_field
    @property
    def report_type(self) -> str:
        return type(self).__name__

    @model_serializer(mode="wrap")
    def _flatten_summary(self, handler) -> dict:
        data = handler(self)
        summary = data.pop("summary", None)
        if isinstance(summary, dict):
            data = {**summary, **data}
        return data

    @abc.abstractmethod
    def render(self) -> None: ...

    def to_dict(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)

    def save(self, path: str | Path) -> Path:
        from judgearena.repro import _to_jsonable  # lazy: avoid an import cycle

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_to_jsonable(self.to_dict()), indent=2) + "\n")
        return p


class BattleReport(Report):
    """Pairwise battle results for the arena and MT-Bench pipelines."""

    task: str
    """Evaluation task name."""
    model_a: str = Field(serialization_alias="model_A")
    """Model in the A position."""
    model_b: str = Field(serialization_alias="model_B")
    """Model in the B position."""
    judge_model: str
    """LLM judge that scored the battles."""
    summary: PrefSummary
    """Win/loss/tie statistics (flattened to the top level on serialization)."""
    swap_mode: str | None = None
    """Position-bias handling: "fixed" or "both"."""
    result_folder: str | None = None
    """Directory the run's artifacts were written to."""
    per_category: dict | None = None
    """Per-category win/loss/tie breakdown (MT-Bench)."""
    per_turn: dict | None = None
    """Per-turn win/loss/tie breakdown (MT-Bench)."""
    preferences: list = Field(default_factory=list)
    """Raw per-battle preference values (0=A, 0.5=tie, 1=B)."""
    metadata: dict = Field(default_factory=dict)
    """Free-form run metadata (baseline assignment, prompt preset, ...)."""

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
