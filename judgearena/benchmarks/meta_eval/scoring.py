"""Agreement and ranking metrics for judge meta-evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from judgearena.benchmarks.elo.rating import fit_bradley_terry
from judgearena.benchmarks.meta_eval.sampling import comparison_components

_REQUIRED_COLUMNS = {
    "battle_id",
    "model_a",
    "model_b",
    "reference_pref",
    "pref",
    "sampled",
}
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
    models = pd.concat([battles["model_a"], battles["model_b"]])
    if models.isna().any() or not all(isinstance(model, str) for model in models):
        raise ValueError("Meta-evaluation model names must be non-null strings.")
    if (battles["model_a"] == battles["model_b"]).any():
        raise ValueError("Meta-evaluation battles do not allow self-comparisons.")
    if battles["sampled"].isna().any() or not all(
        isinstance(value, (bool, np.bool_)) for value in battles["sampled"]
    ):
        raise ValueError("Meta-evaluation sampled values must be booleans.")
    if not all(
        not isinstance(value, bool)
        and isinstance(value, Real)
        and float(value) in (0.0, 0.5, 1.0)
        for value in battles["reference_pref"]
    ):
        raise ValueError("Meta-evaluation reference_pref values must be 0, 0.5, or 1.")
    if not all(
        pd.isna(value)
        or (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
        )
        for value in battles["pref"]
    ):
        raise ValueError(
            "Meta-evaluation non-null pref values must be finite numeric preferences "
            "in [0, 1]."
        )


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
    labels = np.unique(np.concatenate((reference, judge)))
    if len(labels) < 2:
        return float("nan")
    return float(cohen_kappa_score(reference, judge, labels=[0, 1, 2]))


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
    complete = rows["pref"].notna()
    complete_rows = rows.loc[complete]
    n_complete = len(complete_rows)
    if n_complete:
        complete_reference = _reference_labels(complete_rows["reference_pref"])
        complete_judge = _hard_preferences(complete_rows["pref"], tie_tolerance)
        complete_correct = complete_reference == complete_judge
        accuracy_complete = float(np.mean(complete_correct))
        kappa = _cohen_kappa(complete_reference, complete_judge)
    else:
        complete_correct = np.array([], dtype=bool)
        accuracy_complete = kappa = float("nan")

    if n_attempted:
        correct = np.zeros(n_attempted, dtype=bool)
        correct[complete.to_numpy()] = complete_correct
        accuracy_attempted = float(np.mean(correct))
        coverage = n_complete / n_attempted
    else:
        accuracy_attempted = coverage = float("nan")
    return {
        "n_attempted": n_attempted,
        "n_complete": n_complete,
        "coverage": coverage,
        "accuracy_attempted": accuracy_attempted,
        "accuracy_complete": accuracy_complete,
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
    complete_samples: list[float] = []
    kappa_samples: list[float] = []
    if len(rows):
        for _ in range(n_bootstraps):
            assert rng is not None
            indices = rng.integers(0, len(rows), size=len(rows))
            sample = rows.iloc[indices]
            values = _agreement_point(sample, tie_tolerance)
            attempted_samples.append(float(values["accuracy_attempted"]))
            complete_accuracy = float(values["accuracy_complete"])
            if math.isfinite(complete_accuracy):
                complete_samples.append(complete_accuracy)
            kappa = float(values["cohen_kappa"])
            if math.isfinite(kappa):
                kappa_samples.append(kappa)
    return {
        **point,
        "accuracy_attempted_se": _sample_std(attempted_samples),
        "accuracy_complete_se": _sample_std(complete_samples),
        "accuracy_complete_bootstraps_valid": len(complete_samples),
        "cohen_kappa_se": _sample_std(kappa_samples),
        "n_bootstraps_requested": n_bootstraps,
        "n_kappa_bootstraps_valid": len(kappa_samples),
    }


@dataclass(frozen=True, kw_only=True)
class MetaEvalAgreementMetric:
    """Configured battle-level agreement with human reference preferences."""

    tie_tolerance: float
    n_bootstraps: int

    def __post_init__(self) -> None:
        _validate_configuration(self.n_bootstraps, self.tie_tolerance)

    def calculate(
        self,
        battles: pd.DataFrame,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        """Calculate attempted and complete agreement views."""
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
                view["accuracy_complete"], view["accuracy_complete_se"]
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
        spearman = float(spearmanr(reference, judge).statistic)
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
    vectors = {
        name: _centered_vector(fit_bradley_terry(fitting, pref_col=name), models)
        for name in ("human", "hard", "soft")
    }
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

    tie_tolerance: float
    include_human_ties: bool
    n_bootstraps: int

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
        complete = battles["sampled"] & battles["pref"].notna()
        rows = battles.loc[complete].copy()
        if not self.include_human_ties:
            human_tie = rows["reference_pref"].eq(0.5)
            rows = rows.loc[~human_tie]

        unavailable = _unavailable_ranking(
            n_battles=len(rows),
            n_models=len(models),
            n_bootstraps=self.n_bootstraps,
        )
        if comparison_components(rows, models) != [frozenset(models)]:
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


_ELO_GAP_METHODS = ("hard", "soft", "hard_no_judge_ties")


def _validate_elo_gap_configuration(
    battle_counts: object, n_seeds: int, tie_tolerance: float
) -> tuple[int, ...]:
    _validate_configuration(0, tie_tolerance)
    if not isinstance(battle_counts, (list, tuple)) or not battle_counts:
        raise ValueError("battle_counts must be a non-empty ordered sequence")
    if any(type(count) is not int or count <= 0 for count in battle_counts):
        raise ValueError("battle_counts values must be positive integers")
    counts = tuple(battle_counts)
    if any(left >= right for left, right in zip(counts, counts[1:], strict=False)):
        raise ValueError("battle_counts values must be unique and ordered ascending")
    if type(n_seeds) is not int or n_seeds <= 0:
        raise ValueError("n_seeds must be a positive integer")
    return counts


def _elo_gap_priority(
    schedule_seed: int, replicate: int, focal_model: str, battle_id: object
) -> bytes:
    battle_type, battle_value = _battle_sort_key(battle_id)
    payload = (
        f"{schedule_seed}\0{replicate}\0{focal_model}\0{battle_type}\0{battle_value}"
    )
    return hashlib.sha256(payload.encode()).digest()


def _elo_gap_vector(rows: pd.DataFrame, models: list[str]) -> np.ndarray | None:
    if comparison_components(rows, models) != [frozenset(models)]:
        return None
    return _centered_vector(fit_bradley_terry(rows, pref_col="pref"), models)


def _elo_gap_rows(
    *,
    models: list[str],
    battles: pd.DataFrame,
    schedules: dict[tuple[int, str], list[object]],
    reference: np.ndarray | None,
    battle_counts: tuple[int, ...],
    n_seeds: int,
    tie_tolerance: float,
) -> dict[str, list[dict[str, float | int]]]:
    results: dict[str, list[dict[str, float | int]]] = {
        variant: [] for variant in _ELO_GAP_METHODS
    }
    by_id = battles.set_index("battle_id", drop=False)
    human_by_model = {}
    for focal_model in models:
        incident = battles["model_a"].eq(focal_model) | battles["model_b"].eq(
            focal_model
        )
        human_by_model[focal_model] = battles.loc[
            ~incident, ["model_a", "model_b", "reference_pref"]
        ].rename(columns={"reference_pref": "pref"})

    for battle_count in battle_counts:
        replicate_gaps = {variant: [] for variant in _ELO_GAP_METHODS}
        complete_counts: list[int] = []
        used_counts = {variant: [] for variant in _ELO_GAP_METHODS}

        for replicate in range(n_seeds):
            gaps = {variant: [] for variant in _ELO_GAP_METHODS}
            for focal_index, focal_model in enumerate(models):
                selected_ids = schedules[replicate, focal_model][:battle_count]
                selected = by_id.loc[selected_ids]
                complete = selected.loc[selected["pref"].notna()].copy()
                hard_prefs = _hard_preferences(complete["pref"], tie_tolerance) / 2.0
                complete_counts.append(len(complete))
                human = human_by_model[focal_model]

                for variant in _ELO_GAP_METHODS:
                    judge = complete[["model_a", "model_b"]].copy()
                    judge["pref"] = (
                        complete["pref"].to_numpy(dtype=float)
                        if variant == "soft"
                        else hard_prefs
                    )
                    if variant == "hard_no_judge_ties":
                        judge = judge.loc[judge["pref"].ne(0.5)]
                    used_counts[variant].append(len(judge))

                    hybrid = pd.concat([human, judge], ignore_index=True)
                    fitted = _elo_gap_vector(hybrid, models)
                    if reference is not None and fitted is not None:
                        gaps[variant].append(
                            abs(fitted[focal_index] - reference[focal_index])
                        )

            for variant in _ELO_GAP_METHODS:
                if len(gaps[variant]) == len(models) and models:
                    replicate_gaps[variant].append(float(np.mean(gaps[variant])))

        for variant in _ELO_GAP_METHODS:
            valid = replicate_gaps[variant]
            mean_gap = float(np.mean(valid)) if valid else float("nan")
            sampling_se = (
                float(np.std(valid, ddof=1) / np.sqrt(len(valid)))
                if len(valid) >= 2
                else float("nan")
            )
            results[variant].append(
                {
                    "attempted_battles_per_model": battle_count,
                    "mean_gap": mean_gap,
                    "sampling_se": sampling_se,
                    "n_seeds_valid": len(valid),
                    "n_seeds_failed": n_seeds - len(valid),
                    "n_models": len(models),
                    "mean_complete_per_model": float(np.mean(complete_counts))
                    if complete_counts
                    else float("nan"),
                    "mean_used_per_model": float(np.mean(used_counts[variant]))
                    if used_counts[variant]
                    else float("nan"),
                }
            )
    return results


@dataclass(frozen=True, kw_only=True)
class MetaEvalEloGapMetric:
    """Held-out focal-model Elo error at fixed annotation budgets."""

    battle_counts: tuple[int, ...]
    n_seeds: int
    tie_tolerance: float

    def __post_init__(self) -> None:
        counts = _validate_elo_gap_configuration(
            self.battle_counts, self.n_seeds, self.tie_tolerance
        )
        object.__setattr__(self, "battle_counts", counts)

    def calculate(
        self,
        battles: pd.DataFrame,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        """Calculate the hard, soft, and judge-tie-excluded Elo-gap methods."""
        _validate_battles(battles)
        if rng is None:
            raise ValueError("Meta-evaluation Elo gap requires an RNG.")

        models = sorted(set(battles["model_a"]) | set(battles["model_b"]))
        attempted = battles.loc[battles["sampled"]]
        maximum = max(self.battle_counts)
        attempted_ids_by_model = {}
        shortfalls = {}
        for model in models:
            incident = attempted["model_a"].eq(model) | attempted["model_b"].eq(model)
            battle_ids = attempted.loc[incident, "battle_id"].tolist()
            attempted_ids_by_model[model] = battle_ids
            if len(battle_ids) < maximum:
                shortfalls[model] = len(battle_ids)
        if shortfalls:
            raise ValueError(
                f"Every model needs at least {maximum} attempted incident battles; "
                f"available counts: {shortfalls}."
            )

        schedule_seed = int(rng.integers(0, 2**63))
        schedules: dict[tuple[int, str], list[object]] = {}
        for replicate in range(self.n_seeds):
            for focal_model in models:
                schedules[replicate, focal_model] = sorted(
                    attempted_ids_by_model[focal_model],
                    key=lambda battle_id: (
                        _elo_gap_priority(
                            schedule_seed, replicate, focal_model, battle_id
                        ),
                        _battle_sort_key(battle_id),
                    ),
                )

        human = battles[["model_a", "model_b", "reference_pref"]].rename(
            columns={"reference_pref": "pref"}
        )
        reference = _elo_gap_vector(human, models)
        methods = _elo_gap_rows(
            models=models,
            battles=battles,
            schedules=schedules,
            reference=reference,
            battle_counts=self.battle_counts,
            n_seeds=self.n_seeds,
            tie_tolerance=self.tie_tolerance,
        )
        return {
            "schedule_seed": schedule_seed,
            "battle_counts_requested": list(self.battle_counts),
            "n_seeds_requested": self.n_seeds,
            **methods,
        }

    @staticmethod
    def render(values: dict[str, object]) -> str:
        """Render Elo gaps and their sampling standard errors."""
        lines = [
            "meta_eval_elo_gap: "
            f"{values['n_seeds_requested']} sampling seeds "
            f"(schedule seed {values['schedule_seed']})"
        ]
        for variant in _ELO_GAP_METHODS:
            lines.append(f"  {variant}:")
            for row in values[variant]:
                gap = _format_estimate(row["mean_gap"], row["sampling_se"], digits=1)
                lines.append(
                    f"    {row['attempted_battles_per_model']} attempted battles/focal: "
                    f"mean_gap={gap} (sampling SE; "
                    f"valid seeds {row['n_seeds_valid']}/"
                    f"{values['n_seeds_requested']}, "
                    f"complete/model={row['mean_complete_per_model']:.1f}, "
                    f"used/model={row['mean_used_per_model']:.1f})"
                )
        return "\n".join(lines)
