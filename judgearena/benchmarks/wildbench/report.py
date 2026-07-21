"""Result reporting for WildBench score and reward runs."""

from __future__ import annotations

from typing import Literal

from judgearena.utils.eval import Report


class WildBenchReport(Report):
    """Typed report shared by WB-Score and WB-Reward runs."""

    task: str
    mode: Literal["score", "reward"]
    model_name: str
    judge_model: str
    baseline_models: list[str]
    num_examples: int
    num_judgments: int
    num_missing: int
    wb_score: float | None = None
    raw_mean_score: float | None = None
    task_macro_score: float | None = None
    wb_reward: float | None = None
    task_macro_reward: float | None = None
    per_category: dict[str, float]
    per_baseline: dict[str, float]
    metadata: dict[str, object]

    def render(self) -> None:
        print(f"\n=== WildBench V2 {self.mode.title()} for {self.model_name} ===")
        if self.mode == "score":
            print(f"WB-Score: {self.wb_score:.2f}")
            print(f"Raw mean score: {self.raw_mean_score:.3f}/10")
            print(f"Task-macro WB-Score: {self.task_macro_score:.2f}")
        else:
            print(f"WB-Reward: {self.wb_reward:.2f}")
            print(f"Task-macro WB-Reward: {self.task_macro_reward:.2f}")
            for baseline, reward in self.per_baseline.items():
                print(f"  {baseline}: {reward:.2f}")
        print(
            f"Examples: {self.num_examples} | Judgments: {self.num_judgments} | "
            f"Missing parses: {self.num_missing}"
        )
        if self.per_category:
            print("Per category:")
            for category, value in sorted(self.per_category.items()):
                print(f"  {category}: {value:.2f}")
