"""WildBench's graded reward and optional character-length penalty."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from judgearena.battles import focal_battles, preference_values
from judgearena.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, kw_only=True)
class WildBenchRewardMetric:
    """Mean reward on [-100, 100]; negative thresholds disable the length penalty."""

    length_penalty_threshold: int = -1

    def __post_init__(self):
        if type(self.length_penalty_threshold) is not int:
            raise ValueError("length_penalty_threshold must be an integer.")

    def calculate(self, battles: pd.DataFrame) -> dict[str, object]:
        frame = focal_battles(battles)
        preferences = preference_values(frame["pref"])
        if not preferences.dropna().isin([0, 0.25, 0.5, 0.75, 1]).all():
            raise ValueError("WildBench reward requires five-level graded preferences.")
        penalized = pd.Series(False, index=frame.index)
        if self.length_penalty_threshold >= 0:
            difference = (
                frame["completion_model"].str.len()
                - frame["completion_baseline"].str.len()
            )
            penalized = (
                preferences.eq(0.25) & difference.gt(self.length_penalty_threshold)
            ) | (preferences.eq(0.75) & difference.lt(-self.length_penalty_threshold))
        # Apply the penalty per judgment, before averaging answer orders.
        rewards = 100 * (1 - 2 * preferences.mask(penalized, 0.5))
        return {
            "reward": float(rewards.mean()) if rewards.notna().any() else None,
            "num_judgments": len(preferences),
            "num_missing": int(preferences.isna().sum()),
            "num_penalized": int(penalized.sum()),
            "length_penalty_threshold": self.length_penalty_threshold,
        }

    @staticmethod
    def render(result: dict[str, object]) -> str:
        reward = result["reward"]
        value = "unavailable" if reward is None else f"{reward:.2f}"
        return (
            f"WB-Reward: {value} (scale -100 to 100; {result['num_missing']} missing)"
        )


class WildBenchRewardMixMetric(WildBenchRewardMetric):
    """Equal mean of per-reference rewards, requiring a result for each reference."""

    def calculate(
        self, battles: pd.DataFrame, *, references: Sequence[str]
    ) -> dict[str, object]:
        frame = (
            battles.dropna(subset=["evaluation_model"])
            if "evaluation_model" in battles
            else battles
        )
        baselines = focal_battles(frame)["baseline"]
        results = {}
        for reference in references:
            results[reference] = super().calculate(frame.loc[baselines == reference])
        missing = [
            reference
            for reference, result in results.items()
            if result["reward"] is None
        ]
        if missing:
            logger.warning(
                "WB-Reward-Mix unavailable: no valid judgments for %s.", missing
            )
        reward = (
            sum(result["reward"] for result in results.values()) / len(results)
            if results and not missing
            else None
        )
        return {"reward": reward, "references": results}

    @staticmethod
    def render(result: dict[str, object]) -> str:
        reward = result["reward"]
        value = "unavailable" if reward is None else f"{reward:.2f}"
        return f"WB-Reward-Mix: {value} (scale -100 to 100)"


@dataclass(frozen=True, kw_only=True)
class WildBenchScoreMetric:
    """Average independent scores, normalized as (mean - 5) * 2."""

    def calculate(self, annotations: pd.DataFrame) -> dict[str, object]:
        scores = pd.to_numeric(annotations["score"], errors="raise")
        if not scores.dropna().between(1, 10).all():
            raise ValueError("WildBench scores must be between 1 and 10.")
        mean = float(scores.mean()) if scores.notna().any() else None
        return {
            "raw_mean": mean,
            "score": None if mean is None else (mean - 5) * 2,
            "num_samples": len(scores),
            "num_missing": int(scores.isna().sum()),
        }

    @staticmethod
    def render(result: dict[str, object]) -> str:
        return f"WB-Score: {result['score']} ({result['num_missing']} missing)"
