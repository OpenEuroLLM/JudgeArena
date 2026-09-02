"""Serializable reports for configured battle metrics."""

from __future__ import annotations

import abc
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, computed_field


class Report(BaseModel, abc.ABC):
    """A metric report that renders, serializes, and saves itself."""

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

    @abc.abstractmethod
    def render(self) -> None: ...

    def to_dict(self) -> dict:
        return self.model_dump(by_alias=True, exclude_none=True)

    def save(self, path: str | Path) -> Path:
        from judgearena.artifacts import to_jsonable

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(to_jsonable(self.to_dict()), indent=2) + "\n")
        return output


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

    def render(self) -> None:
        from judgearena.benchmarks.scoring import render_metrics

        print("\n" + "=" * 60)
        print("🏆 MODEL BATTLE RESULTS 🏆".center(60))
        print(f"📊 Task: {self.task}")
        print(f"🤖 Competitors: Model A: {self.model_a} vs Model B: {self.model_b}")
        print(f"⚖️ Judge: {self.judge_model}")
        print("📈 Metrics:")
        print(render_metrics(self.metrics))
        if self.result_folder:
            print(f"📁 Results: {self.result_folder}")
        print("=" * 60 + "\n")


class EloReport(Report):
    """Configured battle metrics for one model against an arena."""

    arena: str
    judge_model: str
    metrics: dict[str, dict[str, object]]
    num_battles: int
    model_name: str
    sampling_metadata: dict[str, object]

    def render(self) -> None:
        from judgearena.benchmarks.scoring import render_metrics

        print(f"\n=== Results for {self.model_name} ===")
        print(f"Arena: {self.arena} | Judge: {self.judge_model}")
        print(render_metrics(self.metrics))
