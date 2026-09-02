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
    winrate = float((num_wins + 0.5 * num_ties) / denom) if denom else float("nan")
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
        from judgearena.artifacts import to_jsonable  # lazy: avoid an import cycle

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(to_jsonable(self.to_dict()), indent=2) + "\n")
        return p


class BattleReport(Report):
    """Metric results for pairwise and MT-Bench evaluations."""

    task: str
    model_a: str = Field(serialization_alias="model_A")
    model_b: str = Field(serialization_alias="model_B")
    judge_model: str
    metrics: dict[str, dict[str, object]]
    swap_mode: str | None = None
    result_folder: str | None = None
    preferences: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @computed_field
    @property
    def schema_version(self) -> str:
        return "2"

    def render(self) -> None:
        print("\n" + "=" * 60)
        print("🏆 MODEL BATTLE RESULTS 🏆".center(60))
        print(f"📊 Task: {self.task}")
        print(f"🤖 Competitors: Model A: {self.model_a} vs Model B: {self.model_b}")
        print(f"⚖️ Judge: {self.judge_model}")
        print("📈 Metrics:")
        for name, result in self.metrics.items():
            overall = {key: value for key, value in result.items() if key != "groups"}
            print(f"   {name}: {overall}")
            for field, groups in result.get("groups", {}).items():
                for group in groups:
                    print(f"      {field}={group['group']}: {group['values']}")
        if self.result_folder:
            print(f"📁 Results: {self.result_folder}")
        print("=" * 60 + "\n")
