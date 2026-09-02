"""Battle-table scoring functions for Elo tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from judgearena.battles import summarize_bootstrap
from judgearena.benchmarks.elo.rating import fit_bradley_terry


def _anchor_ratings(
    ratings: dict[str, float], baseline_model: str | None
) -> dict[str, float]:
    """Apply the existing optional Elo origin when the anchor is present."""
    if baseline_model is None or baseline_model not in ratings:
        return ratings
    shift = 1000.0 - ratings[baseline_model]
    return {model: rating + shift for model, rating in ratings.items()}


@dataclass(frozen=True, kw_only=True)
class BradleyTerryMetric:
    """Configured arena-anchored Bradley-Terry calculation."""

    n_bootstraps: int = 0
    baseline_model: str | None = None
    soft: bool = True

    def __post_init__(self) -> None:
        if type(self.n_bootstraps) is not int or self.n_bootstraps < 0:
            raise ValueError("n_bootstraps must be a non-negative integer")
        if self.baseline_model is not None and not isinstance(self.baseline_model, str):
            raise TypeError("baseline_model must be a string or None")
        if type(self.soft) is not bool:
            raise TypeError("soft must be a boolean")

    def calculate(
        self,
        battles: pd.DataFrame,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        """Calculate Bradley-Terry ratings from battle rows."""
        required = {"model_a", "model_b", "pref"}
        if not self.soft:
            required.add("pref_hard")
        missing = sorted(required - set(battles.columns))
        if missing:
            raise ValueError(f"Bradley-Terry battles are missing columns: {missing}.")
        scoring_battles = battles.copy()
        if not self.soft:
            scoring_battles["pref"] = scoring_battles["pref_hard"]
        point_ratings = _anchor_ratings(
            fit_bradley_terry(scoring_battles, pref_col="pref"), self.baseline_model
        )
        lifecycle_columns = {"pref_hard", "source", "evaluation_model"}
        missing_lifecycle = sorted(lifecycle_columns - set(battles.columns))
        if missing_lifecycle:
            if self.n_bootstraps > 0:
                raise ValueError(
                    "Bootstrapped Bradley-Terry battles are missing columns: "
                    f"{missing_lifecycle}."
                )
            return {"ratings": point_ratings}

        evaluation_models = battles["evaluation_model"].dropna().unique()
        if len(evaluation_models) != 1:
            raise ValueError(
                "Bradley-Terry requires exactly one model under evaluation."
            )
        evaluation_model = str(evaluation_models[0])
        if self.n_bootstraps > 0 and rng is None:
            raise ValueError("Bootstrapped Bradley-Terry requires an RNG.")

        human_battles = scoring_battles.loc[scoring_battles["source"] == "human"].copy()
        human_battles["pref"] = human_battles["pref_hard"]
        human_ratings = _anchor_ratings(
            fit_bradley_terry(human_battles, pref_col="pref"), self.baseline_model
        )

        battle_counts: dict[str, int] = {}
        for model in pd.concat(
            [scoring_battles["model_a"], scoring_battles["model_b"]]
        ):
            battle_counts[model] = battle_counts.get(model, 0) + 1

        bootstrap_ratings: list[dict[str, float]] = []
        for _ in range(self.n_bootstraps):
            assert rng is not None
            sample = scoring_battles.sample(
                n=len(scoring_battles),
                replace=True,
                random_state=int(rng.integers(0, 2**31)),
            )
            bootstrap_ratings.append(
                _anchor_ratings(
                    fit_bradley_terry(sample, pref_col="pref"), self.baseline_model
                )
            )

        rating_entries = (
            summarize_bootstrap(bootstrap_ratings, battle_counts, evaluation_model)
            if bootstrap_ratings
            else []
        )
        mean_ratings = {entry.model: entry.rating for entry in rating_entries}
        overlap = [
            model
            for model in mean_ratings
            if model in human_ratings and model != evaluation_model
        ]
        mae_vs_human = (
            float(
                np.mean(
                    [
                        abs(mean_ratings[model] - human_ratings[model])
                        for model in overlap
                    ]
                )
            )
            if overlap
            else float("nan")
        )
        model_rating_values = [
            ratings[evaluation_model]
            for ratings in bootstrap_ratings
            if evaluation_model in ratings
        ]

        return {
            "ratings": point_ratings,
            "rating": float(np.mean(model_rating_values))
            if model_rating_values
            else float("nan"),
            "rating_std": float(np.std(model_rating_values))
            if model_rating_values
            else float("nan"),
            "rating_n_bootstraps": len(model_rating_values),
            "mean_ratings": mean_ratings,
            "human_ratings": human_ratings,
            "bootstrap_ratings": bootstrap_ratings,
            "rating_entries": [asdict(entry) for entry in rating_entries],
            "battle_counts": battle_counts,
            "mae_vs_human": mae_vs_human,
            "mae_num_models": len(overlap),
            "n_bootstraps": self.n_bootstraps,
            "method": "Soft-ELO" if self.soft else "ELO",
            "evaluation_model": evaluation_model,
            "llm_judged_battles": int((battles["source"] == "llm-judge").sum()),
            "human_anchor_battles": len(human_battles),
        }

    @staticmethod
    def render(values: dict[str, object]) -> str:
        """Render point or arena-anchored Bradley-Terry values."""
        if "method" not in values:
            lines = ["bradley_terry ratings:"]
            ratings = values["ratings"]
            if not ratings:
                lines.append("  Not enough data to compute ratings.")
            else:
                lines.extend(
                    f"  {model}: {rating:.1f}"
                    for model, rating in sorted(
                        ratings.items(), key=lambda item: -item[1]
                    )
                )
            return "\n".join(lines)
        lines = [
            f"bradley_terry: {values['method']} ratings "
            f"({values['n_bootstraps']} bootstraps)",
            f"{values['llm_judged_battles']} judged battles and "
            f"{values['human_anchor_battles']} human anchor battles",
        ]
        entries = values["rating_entries"]
        if not entries:
            ratings = values["ratings"]
            if ratings:
                lines.append("Point ratings:")
                lines.extend(
                    f"  {model}: {rating:.1f}"
                    for model, rating in sorted(
                        ratings.items(), key=lambda item: -item[1]
                    )
                )
            else:
                lines.append("  Not enough data to compute ratings.")
            return "\n".join(lines)
        evaluation_model = values["evaluation_model"]
        for entry in entries:
            suffix = " <-----" if entry["model"] == evaluation_model else ""
            lines.append(
                f"  {entry['model']} ({entry['n_battles']}){suffix}: "
                f"{entry['rating']:.1f} "
                f"[{entry['ci_low']:.1f}, {entry['ci_high']:.1f}]"
            )
        if values["mae_num_models"]:
            lines.append(
                f"MAE vs Human-ELO ({values['mae_num_models']} arena models): "
                f"{values['mae_vs_human']:.1f}"
            )
        else:
            lines.append("No overlapping arena models to compute MAE.")
        return "\n".join(lines)
