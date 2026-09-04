"""Agreement and ranking metrics for judge meta-evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from judgearena.benchmarks.elo.rating import fit_bradley_terry
from judgearena.benchmarks.meta_eval.sampling import comparison_components

_REQUIRED_COLUMNS = {
    "battle_id",
    "model_a",
    "model_b",
    "reference_pref",
    "pref",
    "sampled",
    "parse_status",
}
_PARSE_STATUSES = {"complete", "partial", "missing"}
_METRIC_NAMES = ("spearman", "elo_mae")


def _validate_configuration(n_bootstraps: int, tie_tolerance: float) -> None:
    if type(n_bootstraps) is not int or n_bootstraps < 0:
        raise ValueError("n_bootstraps must be a non-negative integer")
    if (
        isinstance(tie_tolerance, bool)
        or not isinstance(tie_tolerance, Real)
        or not math.isfinite(float(tie_tolerance))
        or not 0.0 <= float(tie_tolerance) < 0.5
    ):
        raise ValueError("tie_tolerance must be a finite number in [0, 0.5)")


def _validate_battles(battles: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS - set(battles.columns))
    if missing:
        raise ValueError(f"Meta-evaluation battles are missing columns: {missing}.")
    if battles["battle_id"].isna().any() or battles["battle_id"].duplicated().any():
        raise ValueError("Meta-evaluation battle_id values must be present and unique.")
    if battles[["model_a", "model_b"]].isna().any().any():
        raise ValueError("Meta-evaluation model names must not be missing.")
    if not all(
        isinstance(model, str)
        for model in pd.concat([battles["model_a"], battles["model_b"]])
    ):
        raise ValueError("Meta-evaluation model names must be strings.")
    if (battles["model_a"] == battles["model_b"]).any():
        raise ValueError("Meta-evaluation battles do not allow self-comparisons.")
    if battles["sampled"].isna().any() or not all(
        isinstance(value, (bool, np.bool_)) for value in battles["sampled"]
    ):
        raise ValueError("Meta-evaluation sampled values must be booleans.")

    sampled = battles["sampled"]
    sampled_status = battles.loc[sampled, "parse_status"]
    invalid_statuses = set(sampled_status.dropna()) - _PARSE_STATUSES
    if sampled_status.isna().any() or invalid_statuses:
        raise ValueError(
            "Sampled meta-evaluation parse_status values must be complete, partial, "
            f"or missing; got {sorted(map(str, invalid_statuses))}."
        )

    for column, allow_missing in (("reference_pref", False), ("pref", True)):
        invalid = []
        for value in battles[column]:
            if pd.isna(value):
                if not allow_missing:
                    invalid.append(value)
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                invalid.append(value)
        if invalid:
            raise ValueError(
                f"Meta-evaluation {column} values must be numeric preferences in "
                "[0, 1]."
            )

    if not battles["reference_pref"].isin((0.0, 0.5, 1.0)).all():
        raise ValueError("Meta-evaluation reference_pref values must be 0, 0.5, or 1.")
    complete = sampled & battles["parse_status"].eq("complete")
    if battles.loc[complete, "pref"].isna().any():
        raise ValueError("Complete sampled battles must have a judge preference.")
    missing = sampled & battles["parse_status"].eq("missing")
    if battles.loc[missing, "pref"].notna().any():
        raise ValueError("Missing sampled battles cannot have a judge preference.")


def _reference_labels(values: pd.Series) -> np.ndarray:
    return (values.to_numpy(dtype=float) * 2).astype(int)


def _hard_preferences(values: pd.Series, tie_tolerance: float) -> np.ndarray:
    numeric = values.to_numpy(dtype=float)
    return np.where(
        numeric < 0.5 - tie_tolerance,
        0,
        np.where(numeric > 0.5 + tie_tolerance, 2, 1),
    )


def _cohen_kappa(reference: np.ndarray, judge: np.ndarray) -> float:
    if len(reference) == 0:
        return float("nan")
    observed = float(np.mean(reference == judge))
    reference_counts = np.bincount(reference, minlength=3) / len(reference)
    judge_counts = np.bincount(judge, minlength=3) / len(judge)
    expected = float(np.dot(reference_counts, judge_counts))
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan")


def _format_estimate(value: object, se: object, *, digits: int = 3) -> str:
    estimate = float(value)
    uncertainty = float(se)
    if not math.isfinite(estimate):
        return "n/a"
    if not math.isfinite(uncertainty):
        return f"{estimate:.{digits}f}"
    return f"{estimate:.{digits}f} ± {uncertainty:.{digits}f}"


def _battle_sort_key(value: object) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _agreement_point(
    rows: pd.DataFrame, tie_tolerance: float
) -> dict[str, float | int]:
    n_attempted = len(rows)
    complete = rows["parse_status"].eq("complete")
    parsed = rows.loc[complete]
    n_complete = len(parsed)
    if n_complete:
        parsed_reference = _reference_labels(parsed["reference_pref"])
        parsed_judge = _hard_preferences(parsed["pref"], tie_tolerance)
        parsed_correct = parsed_reference == parsed_judge
        accuracy_parsed = float(np.mean(parsed_correct))
        kappa = _cohen_kappa(parsed_reference, parsed_judge)
    else:
        parsed_correct = np.array([], dtype=bool)
        accuracy_parsed = kappa = float("nan")

    if n_attempted:
        correct = np.zeros(n_attempted, dtype=bool)
        correct[complete.to_numpy()] = parsed_correct
        accuracy_attempted = float(np.mean(correct))
        coverage = n_complete / n_attempted
    else:
        accuracy_attempted = coverage = float("nan")
    return {
        "n_attempted": n_attempted,
        "n_complete": n_complete,
        "coverage": coverage,
        "accuracy_attempted": accuracy_attempted,
        "accuracy_parsed": accuracy_parsed,
        "cohen_kappa": kappa,
    }


def _agreement_view(
    rows: pd.DataFrame,
    *,
    tie_tolerance: float,
    n_bootstraps: int,
    rng: np.random.Generator | None,
) -> dict[str, float | int]:
    rows = (
        rows.assign(_battle_sort=rows["battle_id"].map(_battle_sort_key))
        .sort_values("_battle_sort", kind="stable")
        .drop(columns="_battle_sort")
    )
    point = _agreement_point(rows, tie_tolerance)
    attempted_samples: list[float] = []
    parsed_samples: list[float] = []
    kappa_samples: list[float] = []
    if len(rows):
        for _ in range(n_bootstraps):
            assert rng is not None
            indices = rng.integers(0, len(rows), size=len(rows))
            sample = rows.iloc[indices]
            values = _agreement_point(sample, tie_tolerance)
            attempted_samples.append(float(values["accuracy_attempted"]))
            parsed_accuracy = float(values["accuracy_parsed"])
            if math.isfinite(parsed_accuracy):
                parsed_samples.append(parsed_accuracy)
            kappa = float(values["cohen_kappa"])
            if math.isfinite(kappa):
                kappa_samples.append(kappa)
    return {
        **point,
        "accuracy_attempted_se": _sample_std(attempted_samples),
        "accuracy_parsed_se": _sample_std(parsed_samples),
        "accuracy_parsed_bootstraps_valid": len(parsed_samples),
        "cohen_kappa_se": _sample_std(kappa_samples),
        "n_bootstraps_requested": n_bootstraps,
        "n_kappa_bootstraps_valid": len(kappa_samples),
    }


@dataclass(frozen=True, kw_only=True)
class MetaEvalAgreementMetric:
    """Configured battle-level agreement with human reference preferences."""

    tie_tolerance: float = 0.01
    n_bootstraps: int = 1000

    def __post_init__(self) -> None:
        _validate_configuration(self.n_bootstraps, self.tie_tolerance)

    def calculate(
        self,
        battles: pd.DataFrame,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        """Calculate attempted and parsed-complete agreement views."""
        _validate_battles(battles)
        if self.n_bootstraps and rng is None:
            raise ValueError("Bootstrapped meta-evaluation agreement requires an RNG.")
        attempted = battles.loc[battles["sampled"]].copy()
        reference_is_tie = attempted["reference_pref"].eq(0.5)
        return {
            "all": _agreement_view(
                attempted,
                tie_tolerance=self.tie_tolerance,
                n_bootstraps=self.n_bootstraps,
                rng=rng,
            ),
            "no_human_ties": _agreement_view(
                attempted.loc[~reference_is_tie],
                tie_tolerance=self.tie_tolerance,
                n_bootstraps=self.n_bootstraps,
                rng=rng,
            ),
        }

    @staticmethod
    def render(values: dict[str, object]) -> str:
        """Render the two agreement views."""
        lines = ["meta_eval_agreement:"]
        for name in ("all", "no_human_ties"):
            view = values[name]
            attempted = _format_estimate(
                view["accuracy_attempted"], view["accuracy_attempted_se"]
            )
            complete = _format_estimate(
                view["accuracy_parsed"], view["accuracy_parsed_se"]
            )
            kappa = _format_estimate(view["cohen_kappa"], view["cohen_kappa_se"])
            lines.append(
                f"  {name} (complete {view['n_complete']}/{view['n_attempted']}): "
                f"accuracy={attempted}, complete_accuracy={complete}, "
                f"complete_kappa={kappa}, complete_coverage={view['coverage']:.3f}"
            )
        return "\n".join(lines)


def _unavailable_ranking(
    *, n_battles: int, n_models: int, n_bootstraps: int
) -> dict[str, object]:
    unavailable = {
        "spearman": float("nan"),
        "spearman_se": float("nan"),
        "spearman_bootstraps_valid": 0,
        "elo_mae": float("nan"),
        "elo_mae_se": float("nan"),
        "elo_mae_bootstraps_valid": 0,
    }
    return {
        "n_battles": n_battles,
        "n_models": n_models,
        "n_bootstraps_requested": n_bootstraps,
        "n_bootstraps_valid": 0,
        "hard": dict(unavailable),
        "soft": dict(unavailable),
    }


def _centered_vector(ratings: dict[str, float], models: list[str]) -> np.ndarray | None:
    if set(ratings) != set(models):
        return None
    vector = np.asarray([ratings[model] for model in models], dtype=float)
    if not np.isfinite(vector).all():
        return None
    return vector - vector.mean()


def _ranking_values(reference: np.ndarray, judge: np.ndarray) -> dict[str, float]:
    if len(np.unique(reference)) < 2 or len(np.unique(judge)) < 2:
        spearman = float("nan")
    else:
        reference_ranks = pd.Series(reference).rank(method="average").to_numpy()
        judge_ranks = pd.Series(judge).rank(method="average").to_numpy()
        spearman = float(np.corrcoef(reference_ranks, judge_ranks)[0, 1])
    return {
        "spearman": spearman,
        "elo_mae": float(np.mean(np.abs(reference - judge))),
    }


def _fit_rating_bundle(
    rows: pd.DataFrame, models: list[str], tie_tolerance: float
) -> dict[str, dict[str, float]] | None:
    fitting = rows[["model_a", "model_b"]].copy()
    fitting["human"] = rows["reference_pref"].to_numpy(dtype=float)
    fitting["hard"] = _hard_preferences(rows["pref"], tie_tolerance) / 2.0
    fitting["soft"] = rows["pref"].to_numpy(dtype=float)
    try:
        vectors = {
            name: _centered_vector(fit_bradley_terry(fitting, pref_col=name), models)
            for name in ("human", "hard", "soft")
        }
    except (TypeError, ValueError):
        return None
    if any(vector is None for vector in vectors.values()):
        return None
    human = vectors["human"]
    assert human is not None
    return {
        name: _ranking_values(human, vector)
        for name, vector in (("hard", vectors["hard"]), ("soft", vectors["soft"]))
        if vector is not None
    }


def _pair_stratified_sample(
    rows: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    working = rows.assign(
        _pair=[
            tuple(sorted(pair))
            for pair in zip(rows["model_a"], rows["model_b"], strict=True)
        ],
        _battle_sort=rows["battle_id"].map(_battle_sort_key),
    ).sort_values(["_pair", "_battle_sort"], kind="stable")
    parts = []
    for _, stratum in working.groupby("_pair", sort=True):
        indices = rng.integers(0, len(stratum), size=len(stratum))
        parts.append(stratum.iloc[indices])
    return pd.concat(parts, ignore_index=True).drop(columns=["_pair", "_battle_sort"])


@dataclass(frozen=True, kw_only=True)
class MetaEvalRankingMetric:
    """Configured hard and soft Bradley-Terry agreement with human rankings."""

    tie_tolerance: float = 0.01
    include_human_ties: bool = False
    n_bootstraps: int = 1000

    def __post_init__(self) -> None:
        _validate_configuration(self.n_bootstraps, self.tie_tolerance)
        if type(self.include_human_ties) is not bool:
            raise TypeError("include_human_ties must be a boolean")

    def calculate(
        self,
        battles: pd.DataFrame,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        """Fit and bootstrap human, hard-judge, and soft-judge rankings."""
        _validate_battles(battles)
        if self.n_bootstraps and rng is None:
            raise ValueError("Bootstrapped meta-evaluation ranking requires an RNG.")
        models = sorted(set(battles["model_a"]) | set(battles["model_b"]))
        complete = battles["sampled"] & battles["parse_status"].eq("complete")
        rows = battles.loc[complete].copy()
        if not self.include_human_ties:
            human_tie = rows["reference_pref"].eq(0.5)
            rows = rows.loc[~human_tie]

        unavailable = _unavailable_ranking(
            n_battles=len(rows),
            n_models=len(models),
            n_bootstraps=self.n_bootstraps,
        )
        if len(models) < 3 or comparison_components(rows, models) != [
            frozenset(models)
        ]:
            return unavailable
        point = _fit_rating_bundle(rows, models, self.tie_tolerance)
        if point is None:
            return unavailable

        samples: list[dict[str, dict[str, float]]] = []
        for _ in range(self.n_bootstraps):
            assert rng is not None
            sample = _pair_stratified_sample(rows, rng)
            fitted = _fit_rating_bundle(sample, models, self.tie_tolerance)
            if fitted is not None:
                samples.append(fitted)

        result = unavailable
        result["n_bootstraps_valid"] = len(samples)
        for kind in ("hard", "soft"):
            result[kind] = dict(point[kind])
            for metric in _METRIC_NAMES:
                valid = [
                    sample[kind][metric]
                    for sample in samples
                    if math.isfinite(sample[kind][metric])
                ]
                result[kind][f"{metric}_se"] = _sample_std(valid)
                result[kind][f"{metric}_bootstraps_valid"] = len(valid)
        return result

    @staticmethod
    def render(values: dict[str, object]) -> str:
        """Render hard and soft ranking summaries."""
        lines = [
            "meta_eval_ranking: "
            f"{values['n_battles']} battles, {values['n_models']} models"
        ]
        for name in ("hard", "soft"):
            ranking = values[name]
            spearman = _format_estimate(ranking["spearman"], ranking["spearman_se"])
            elo_mae = _format_estimate(
                ranking["elo_mae"], ranking["elo_mae_se"], digits=1
            )
            lines.append(f"  {name}: spearman={spearman}, elo_mae={elo_mae}")
        return "\n".join(lines)
