"""Official WildBench V2 aggregation functions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

TASK_WEIGHTS = {
    "Creative Tasks": 0.5,
    "Planning & Reasoning": 1.25,
    "Math & Data Analysis": 1.0,
    "Information/Advice seeking": 0.75,
    "Coding & Debugging": 1.25,
}

WildBenchScorer = Callable[[pd.DataFrame, pd.DataFrame], dict[str, object]]


def _published_score(raw_score: float) -> float:
    return (raw_score - 5.0) * 2.0


def score_wildbench_v2(
    examples: pd.DataFrame, annotations: pd.DataFrame
) -> dict[str, object]:
    """Aggregate WB-Score exactly as the official V2 leaderboard scripts."""
    valid = annotations.dropna(subset=["score"])
    raw_mean = float(valid["score"].mean()) if not valid.empty else float("nan")

    categories_by_session = examples.set_index("instruction_index")["task_categories"]
    category_scores: dict[str, list[float]] = {}
    for row in valid.itertuples(index=False):
        for category in categories_by_session.loc[str(row.session_id)]:
            category_scores.setdefault(category, []).append(float(row.score))
    per_category = {
        category: _published_score(float(np.mean(scores)))
        for category, scores in category_scores.items()
    }

    # Upstream divides by the complete five-category weight sum. This is
    # observable on subsets, so keep it rather than re-normalizing present groups.
    task_macro = sum(
        per_category.get(category, 0.0) * weight
        for category, weight in TASK_WEIGHTS.items()
    ) / sum(TASK_WEIGHTS.values())
    wb_score = _published_score(raw_mean)
    return {
        "num_examples": int(len(annotations)),
        "num_scored": int(len(valid)),
        "num_missing": int(len(annotations) - len(valid)),
        "raw_mean_score": raw_mean,
        "wb_score": wb_score,
        "wb_score_leaderboard": wb_score * 10.0,
        "task_macro_score": task_macro,
        "task_macro_score_leaderboard": task_macro * 10.0,
        "per_category": per_category,
    }


WILDBENCH_SCORERS: dict[str, WildBenchScorer] = {
    "wildbench-score-v2": score_wildbench_v2,
}


def resolve_wildbench_scorer(name: str) -> WildBenchScorer:
    """Resolve the scorer selected by a WildBench task definition."""
    try:
        return WILDBENCH_SCORERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown WildBench scorer {name!r}; available: {sorted(WILDBENCH_SCORERS)}"
        ) from exc
