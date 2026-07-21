"""Versioned parsers and official metric implementations for WildBench V2."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from judgearena.utils import strip_thinking_tags

WILDBENCH_V2_TASK_WEIGHTS = {
    "Creative Tasks": 0.5,
    "Planning & Reasoning": 1.25,
    "Math & Data Analysis": 1.0,
    "Information/Advice seeking": 0.75,
    "Coding & Debugging": 1.25,
}

_CHOICE_TO_A_REWARD = {
    "A++": 1.0,
    "A+": 0.5,
    "A=B": 0.0,
    "B+": -0.5,
    "B++": -1.0,
}


@dataclass(frozen=True)
class WildBenchMetrics:
    """Normalized aggregate returned by either WildBench scoring mode."""

    primary_metric: str
    value: float
    task_macro: float
    raw_mean: float | None
    per_category: dict[str, float]
    per_baseline: dict[str, float]


class WildBenchScorer(Protocol):
    """Runtime scorer selected by a WildBench task definition."""

    name: str
    mode: Literal["score", "reward"]
    primary_metric: str

    def parse(self, judge_output: str) -> float | str | None: ...

    def aggregate(
        self,
        examples: pd.DataFrame,
        annotations: pd.DataFrame,
        *,
        baseline_models: list[str],
    ) -> WildBenchMetrics: ...


def _parse_json_object(text: str) -> dict[str, object] | None:
    cleaned = strip_thinking_tags(text).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, json.JSONDecodeError):
        pass

    decoder = json.JSONDecoder()
    for start in [i for i, char in enumerate(cleaned) if char == "{"]:
        try:
            parsed, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_wildbench_score(text: str) -> float | None:
    """Parse an official WB-Score JSON response, including loose JSON output."""
    payload = _parse_json_object(text)
    value = payload.get("score") if payload is not None else None
    if value is None:
        matches = re.findall(
            r'["\']?score["\']?\s*:\s*["\']?(-?\d+(?:\.\d+)?)',
            strip_thinking_tags(text),
            flags=re.IGNORECASE,
        )
        value = matches[-1] if matches else None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if np.isfinite(score) and 1 <= score <= 10 else None


def parse_wildbench_choice(text: str) -> str | None:
    """Parse an official WB-Reward pairwise choice."""
    payload = _parse_json_object(text)
    value = payload.get("choice") if payload is not None else None
    if isinstance(value, str):
        normalized = value.strip().upper().replace(" ", "")
        if normalized in _CHOICE_TO_A_REWARD:
            return normalized
    matches = re.findall(r"A\+\+|B\+\+|A=B|A\+|B\+", strip_thinking_tags(text))
    return matches[-1] if matches else None


def choice_to_candidate_reward(choice: str, *, candidate_is_a: bool) -> float:
    """Convert an A/B judgment into reward from the evaluated model's view."""
    reward_a = _CHOICE_TO_A_REWARD[choice]
    return reward_a if candidate_is_a else -reward_a


def apply_wildbench_length_penalty(
    reward: float,
    candidate_output: str,
    baseline_output: str,
    length_penalty_chars: int | None,
) -> float:
    """Convert only a length-advantaged slight win or loss into a tie."""
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


def _weighted_task_macro(per_category: dict[str, float]) -> float:
    weighted = [
        (value, WILDBENCH_V2_TASK_WEIGHTS[category])
        for category, value in per_category.items()
        if category in WILDBENCH_V2_TASK_WEIGHTS and np.isfinite(value)
    ]
    if not weighted:
        return float("nan")
    return float(
        sum(value * weight for value, weight in weighted)
        / sum(weight for _, weight in weighted)
    )


def _categories_for(examples: pd.DataFrame, session_id: str) -> list[str]:
    categories = examples.loc[session_id, "task_categories"]
    return categories if isinstance(categories, list) else []


@dataclass(frozen=True)
class WildBenchScoreV2:
    """Official WB-Score V2 parser and aggregation."""

    name: str = "wildbench-score-v2"
    mode: Literal["score"] = "score"
    primary_metric: str = "wb_score"

    def parse(self, judge_output: str) -> float | None:
        return parse_wildbench_score(judge_output)

    def aggregate(
        self,
        examples: pd.DataFrame,
        annotations: pd.DataFrame,
        *,
        baseline_models: list[str],
    ) -> WildBenchMetrics:
        del baseline_models
        valid = annotations.dropna(subset=["score"])
        raw_mean = float(valid["score"].mean()) if not valid.empty else float("nan")
        category_values: dict[str, list[float]] = {}
        for row in valid.itertuples(index=False):
            for category in _categories_for(examples, row.session_id):
                category_values.setdefault(category, []).append(float(row.score))
        per_category = {
            category: (float(np.mean(values)) - 5.0) * 20.0
            for category, values in category_values.items()
        }
        value = (raw_mean - 5.0) * 20.0
        return WildBenchMetrics(
            primary_metric=self.primary_metric,
            value=value,
            task_macro=_weighted_task_macro(per_category),
            raw_mean=raw_mean,
            per_category=per_category,
            per_baseline={},
        )


@dataclass(frozen=True)
class WildBenchRewardV2:
    """Official WB-Reward V2 parser and aggregation."""

    name: str = "wildbench-reward-v2"
    mode: Literal["reward"] = "reward"
    primary_metric: str = "wb_reward"

    def parse(self, judge_output: str) -> str | None:
        return parse_wildbench_choice(judge_output)

    def aggregate(
        self,
        examples: pd.DataFrame,
        annotations: pd.DataFrame,
        *,
        baseline_models: list[str],
    ) -> WildBenchMetrics:
        canonical = (
            annotations.groupby(
                ["session_id", "baseline_model"], as_index=False, sort=False
            )["reward"]
            .mean()
            .reset_index(drop=True)
        )
        per_baseline = {}
        for baseline in baseline_models:
            values = canonical.loc[
                canonical["baseline_model"] == baseline, "reward"
            ].dropna()
            per_baseline[baseline] = (
                float(values.mean()) * 100.0 if not values.empty else float("nan")
            )
        valid_baselines = [
            value for value in per_baseline.values() if np.isfinite(value)
        ]
        value = float(np.mean(valid_baselines)) if valid_baselines else float("nan")

        category_baseline_values: dict[str, dict[str, list[float]]] = {}
        for row in canonical.dropna(subset=["reward"]).itertuples(index=False):
            for category in _categories_for(examples, row.session_id):
                category_baseline_values.setdefault(category, {}).setdefault(
                    row.baseline_model, []
                ).append(float(row.reward))
        per_category = {}
        for category, values_by_baseline in category_baseline_values.items():
            baseline_means = [
                float(np.mean(values))
                for values in values_by_baseline.values()
                if values
            ]
            per_category[category] = float(np.mean(baseline_means)) * 100.0
        return WildBenchMetrics(
            primary_metric=self.primary_metric,
            value=value,
            task_macro=_weighted_task_macro(per_category),
            raw_mean=None,
            per_category=per_category,
            per_baseline=per_baseline,
        )


_WILDBENCH_SCORERS: dict[str, WildBenchScorer] = {
    "wildbench-score-v2": WildBenchScoreV2(),
    "wildbench-reward-v2": WildBenchRewardV2(),
}

WILDBENCH_SCORER_NAMES = frozenset(_WILDBENCH_SCORERS)


def resolve_wildbench_scorer(name: str) -> WildBenchScorer:
    """Return the registered scorer selected by a WildBench task."""
    try:
        return _WILDBENCH_SCORERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown WildBench scorer {name!r}; available: "
            f"{sorted(WILDBENCH_SCORER_NAMES)}"
        ) from exc
