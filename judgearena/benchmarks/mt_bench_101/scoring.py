"""Absolute-score metric for MT-Bench-101 single-answer grading."""

from __future__ import annotations

import pandas as pd


def _macro_task_average(battles: pd.DataFrame, score_column: str) -> float | None:
    per_task = battles.groupby("task")[score_column].mean()
    score = per_task.mean()
    return float(score) if pd.notna(score) else None


class MTBench101AbsoluteScoreMetric:
    """Summarize each model's minimum-per-dialogue absolute ratings."""

    def calculate(self, battles: pd.DataFrame, **runtime: object) -> dict[str, object]:
        del runtime
        return {
            "model_A_score": _macro_task_average(battles, "score_A"),
            "model_B_score": _macro_task_average(battles, "score_B"),
            "num_dialogues": int(len(battles)),
            "num_scored_A": int(battles["score_A"].notna().sum()),
            "num_scored_B": int(battles["score_B"].notna().sum()),
        }

    @staticmethod
    def render(result: dict[str, object]) -> str:
        return (
            "MT-Bench-101 absolute score: "
            f"Model A={result['model_A_score']}, Model B={result['model_B_score']}"
        )
