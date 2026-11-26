"""Battle-table scoring functions for Elo tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from judgearena.battles import hard_preferences, has_support, summarize_bootstrap
from judgearena.benchmarks.elo.rating import fit_bradley_terry, fit_frozen_rating
from judgearena.benchmarks.meta_eval.sampling import require_connected_pool


@dataclass(frozen=True, kw_only=True)
class BradleyTerryMetric:
    """Configured arena-anchored Bradley-Terry calculation."""

    n_bootstraps: int = 0
    baseline_model: str | None = None
    soft: bool = True
    fit_mode: Literal["joint", "frozen"] = "joint"
    reference_min_battles: int = 0
    bootstrap_group: str = "question_id"

    def __post_init__(self) -> None:
        if (
            type(self.reference_min_battles) is not int
            or self.reference_min_battles < 0
        ):
            raise ValueError("reference_min_battles must be a non-negative integer")
        if self.fit_mode not in {"joint", "frozen"}:
            raise ValueError("fit_mode must be frozen or joint")
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
        seed: int = 0,
        reference_ratings: pd.DataFrame | None = None,
        reference_battles: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        """Fit the candidate from judged battles and any supplied reference labels."""
        if reference_battles is not None:
            battles = pd.concat([battles, reference_battles], ignore_index=True)
        if self.reference_min_battles and "evaluation_model" in battles:
            is_reference = battles.evaluation_model.isna().to_numpy()
            keep = ~is_reference
            keep[is_reference] = has_support(
                battles.loc[is_reference], self.reference_min_battles
            ).to_numpy()
            battles = battles.loc[keep]
        required = {"model_a", "model_b", "pref"}
        if not self.soft:
            required.add("pref_hard")
        missing = sorted(required - set(battles.columns))
        if missing:
            raise ValueError(f"Bradley-Terry battles are missing columns: {missing}.")
        scoring_battles = battles.copy()
        scoring_battles["rating_pref"] = battles["pref" if self.soft else "pref_hard"]
        if {"reference_pref", "evaluation_model"} <= set(battles.columns):
            reference_rows = (
                battles.evaluation_model.isna() & battles.reference_pref.notna()
            )
            labels = battles.loc[reference_rows, "reference_pref"]
            scoring_battles.loc[reference_rows, "rating_pref"] = (
                labels if self.soft else hard_preferences(labels)
            )
        if self.fit_mode == "frozen":
            return self._calculate_frozen(
                scoring_battles, seed=seed, reference_ratings=reference_ratings
            )
        point_ratings = fit_bradley_terry(
            scoring_battles, pref_col="rating_pref", baseline_model=self.baseline_model
        )
        lifecycle_columns = {"pref_hard", "evaluation_model"}
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
        rng = np.random.default_rng(seed)

        reference_battles = scoring_battles.reindex(
            columns=["model_a", "model_b", "reference_pref"]
        ).dropna(subset=["reference_pref"])
        reference_battles["rating_pref"] = hard_preferences(
            reference_battles["reference_pref"]
        )
        reference_ratings = fit_bradley_terry(
            reference_battles,
            pref_col="rating_pref",
            baseline_model=self.baseline_model,
        )

        battle_counts: dict[str, int] = {}
        for model in pd.concat(
            [scoring_battles["model_a"], scoring_battles["model_b"]]
        ):
            battle_counts[model] = battle_counts.get(model, 0) + 1

        bootstrap_ratings: list[dict[str, float]] = []
        for _ in range(self.n_bootstraps):
            sample = scoring_battles.sample(
                n=len(scoring_battles),
                replace=True,
                random_state=int(rng.integers(0, 2**31)),
            )
            bootstrap_ratings.append(
                fit_bradley_terry(
                    sample, pref_col="rating_pref", baseline_model=self.baseline_model
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
            if model in reference_ratings and model != evaluation_model
        ]
        mae_vs_reference = (
            float(
                np.mean(
                    [
                        abs(mean_ratings[model] - reference_ratings[model])
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
            "reference_ratings": reference_ratings,
            "bootstrap_ratings": bootstrap_ratings,
            "rating_entries": [asdict(entry) for entry in rating_entries],
            "battle_counts": battle_counts,
            "mae_vs_reference": mae_vs_reference,
            "mae_num_models": len(overlap),
            "n_bootstraps": self.n_bootstraps,
            "method": "Soft-ELO" if self.soft else "ELO",
            "evaluation_model": evaluation_model,
            "llm_judged_battles": int(battles["evaluation_model"].notna().sum()),
            "reference_anchor_battles": len(reference_battles),
        }

    def _calculate_frozen(
        self,
        battles: pd.DataFrame,
        *,
        seed: int,
        reference_ratings: pd.DataFrame | None = None,
    ) -> dict:
        """Fit the candidate against saved per-language ratings, or ones fitted here."""
        candidates = battles["evaluation_model"].dropna().unique()
        if len(candidates) != 1:
            raise ValueError("Frozen Elo requires exactly one model under evaluation.")
        candidate = str(candidates[0])
        judgments = battles.loc[battles.evaluation_model.notna()]

        def fit(frame, references):
            return fit_frozen_rating(
                frame,
                candidate,
                references,
                pref_col="rating_pref",
                group_col=self.bootstrap_group,
                n_bootstraps=self.n_bootstraps,
                seed=seed,
            )

        if reference_ratings is not None:
            groups = [
                {
                    "group": language,
                    "values": fit(
                        frame,
                        reference_ratings.loc[reference_ratings.lang.eq(language)]
                        .set_index("model")
                        .rating.to_dict(),
                    ),
                }
                for language, frame in judgments.groupby("lang", sort=True)
            ]
            return {"groups": {"lang": groups}}
        reference_battles = battles.loc[battles.evaluation_model.isna()].dropna(
            subset=["rating_pref"]
        )
        require_connected_pool(reference_battles, (), context="Frozen Elo reference")
        references = fit_bradley_terry(
            reference_battles,
            pref_col="rating_pref",
            baseline_model=self.baseline_model,
        )
        values = fit(judgments, references)
        ratings = {**references, candidate: values["rating"]}
        counts = pd.concat([battles.model_a, battles.model_b]).value_counts()
        return {
            **values,
            "ratings": ratings,
            "rating_entries": [
                {
                    "model": model,
                    "rating": rating,
                    "ci_low": values["ci_low"] if model == candidate else rating,
                    "ci_high": values["ci_high"] if model == candidate else rating,
                    "n_battles": int(counts[model]),
                    "source": "evaluated" if model == candidate else "reference",
                }
                for model, rating in ratings.items()
            ],
            "reference_ratings": references,
            "evaluation_model": candidate,
            "fit_mode": "frozen",
            "intervals": "conditional on frozen references",
        }

    @staticmethod
    def render(values: dict[str, object]) -> str:
        """Render point or arena-anchored Bradley-Terry values."""
        if "rating" in values and "ratings" not in values:
            return (
                f"Rating: {values['rating']:.1f} "
                f"[{values['ci_low']:.1f}, {values['ci_high']:.1f}] "
                "(conditional on frozen references)"
            )
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
            if values.get("fit_mode") == "frozen":
                lines.append(
                    f"{values['evaluation_model']}: {values['rating']:.1f} "
                    f"[{values['ci_low']:.1f}, {values['ci_high']:.1f}] "
                    "(conditional on frozen references)"
                )
            return "\n".join(lines)
        lines = [
            f"bradley_terry: {values['method']} ratings "
            f"({values['n_bootstraps']} bootstraps)",
            f"{values['llm_judged_battles']} judged battles and "
            f"{values['reference_anchor_battles']} reference anchor battles",
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
                f"MAE vs reference ELO ({values['mae_num_models']} arena models): "
                f"{values['mae_vs_reference']:.1f}"
            )
        else:
            lines.append("No overlapping arena models to compute MAE.")
        return "\n".join(lines)


def configure_frozen_metrics(cfg, configured):
    """Resolve rating defaults and require a fixed, language-specific reference scale."""
    resolved = []
    for request, metric in configured:
        if isinstance(metric, BradleyTerryMetric):
            if request.breakdown_by not in ((), ("lang",)):
                raise ValueError("Frozen ratings are grouped by lang")
            if not metric.baseline_model:
                raise ValueError("Set the metric's baseline_model")
        resolved.append((request, metric))
    return tuple(resolved)


def frozen_metric_inputs(requests, *, ratings, methods):
    """Bind saved reference ratings to their metric without changing judging."""
    from judgearena.benchmarks.scoring import build_metrics

    overrides = {
        method["name"]: {
            "fit_mode": "frozen",
            "bootstrap_group": "split_group",
            "n_bootstraps": method["n_bootstraps"],
        }
        for method in methods
    }
    runtime = {
        name: {"reference_ratings": ratings.loc[ratings.method.eq(name)]}
        for name in overrides
    }
    # The frozen fitter already returns per-language groups.
    requests = [
        request.model_copy(update={"breakdown_by": ()})
        if (request.name or request.metric) in overrides
        else request
        for request in requests
    ]
    return build_metrics(requests, parameter_overrides_by_metric=overrides), runtime
