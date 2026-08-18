"""Official WildBench V2 aggregation functions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from judgearena.benchmarks.wildbench.parsing import REWARD_FOR_A
from judgearena.benchmarks.wildbench.prompting import EMPTY_RESPONSE

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

    category_valid = valid
    if "model_output" in category_valid:
        outputs = category_valid["model_output"].fillna("").astype(str)
        category_valid = category_valid.loc[
            outputs.str.len().gt(0) & ~outputs.str.endswith("... (truncated)")
        ]
    categories_by_session = examples.set_index("instruction_index")["task_categories"]
    category_scores: dict[str, list[float]] = {}
    for row in category_valid.itertuples(index=False):
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


def candidate_reward(choice: str, *, candidate_is_a: bool) -> float:
    """Return the graded reward from the evaluated model's orientation."""
    reward_a = REWARD_FOR_A[choice]
    return reward_a if candidate_is_a else -reward_a


def apply_length_penalty(
    reward: float,
    candidate_output: str,
    baseline_output: str,
    length_penalty_chars: int | None,
) -> float:
    """Convert a length-advantaged slight win or loss into a tie."""
    if length_penalty_chars is None or abs(reward) != 0.5:
        return reward
    if (
        reward > 0
        and len(candidate_output) > len(baseline_output) + length_penalty_chars
    ):
        return 0.0
    if (
        reward < 0
        and len(baseline_output) > len(candidate_output) + length_penalty_chars
    ):
        return 0.0
    return reward


def score_wildbench_reward_v2(
    examples: pd.DataFrame, annotations: pd.DataFrame
) -> dict[str, object]:
    """Aggregate the official three-reference WB-Reward-Mix."""
    valid = annotations.dropna(subset=["reward"])
    baselines = list(dict.fromkeys(annotations["baseline_model"].astype(str)))

    eligible = valid
    candidate_outputs = eligible["candidate_output"].fillna("").astype(str)
    baseline_outputs = eligible["baseline_output"].fillna("").astype(str)
    eligible = eligible.loc[
        candidate_outputs.ne(EMPTY_RESPONSE)
        & baseline_outputs.ne(EMPTY_RESPONSE)
        & ~candidate_outputs.str.endswith("... (truncated)")
        & ~baseline_outputs.str.endswith("... (truncated)")
    ]

    per_baseline: dict[str, float] = {}
    for baseline in baselines:
        denominator = int((valid["baseline_model"] == baseline).sum())
        numerator = eligible.loc[eligible["baseline_model"] == baseline, "reward"].sum()
        per_baseline[baseline] = (
            float(numerator / denominator) * 100.0 if denominator else float("nan")
        )

    categories_by_session = examples.set_index("instruction_index")["task_categories"]
    category_baseline_rewards: dict[str, dict[str, list[float]]] = {}
    for row in eligible.itertuples(index=False):
        for category in categories_by_session.loc[str(row.session_id)]:
            category_baseline_rewards.setdefault(category, {}).setdefault(
                str(row.baseline_model), []
            ).append(float(row.reward))
    per_category: dict[str, float] = {}
    for category, rewards_by_baseline in category_baseline_rewards.items():
        baseline_means = [
            float(np.mean(rewards_by_baseline[baseline]))
            for baseline in baselines
            if rewards_by_baseline.get(baseline)
        ]
        per_category[category] = float(np.mean(baseline_means)) * 100.0

    task_macro = sum(
        per_category.get(category, 0.0) * weight
        for category, weight in TASK_WEIGHTS.items()
    ) / sum(TASK_WEIGHTS.values())
    baseline_values = [value for value in per_baseline.values() if np.isfinite(value)]
    return {
        "num_annotations": int(len(annotations)),
        "num_scored": int(len(valid)),
        "num_missing": int(len(annotations) - len(valid)),
        "wb_reward": (
            float(np.mean(baseline_values)) if baseline_values else float("nan")
        ),
        "task_macro_reward": task_macro,
        "per_baseline": per_baseline,
        "per_category": per_category,
    }


WILDBENCH_SCORERS: dict[str, WildBenchScorer] = {
    "wildbench-reward-v2": score_wildbench_reward_v2,
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
