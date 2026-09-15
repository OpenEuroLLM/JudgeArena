"""Reusable metric functions for canonical pairwise battles."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from judgearena.utils.eval import compute_pref_summary

BOOTSTRAP_ROUNDS = 1000
BOOTSTRAP_SEED = 0
CONFIDENCE_LEVEL = 0.95


def _preferences(battles: pd.DataFrame, column: str = "pref") -> pd.Series:
    if column not in battles:
        raise ValueError(f"Pairwise metrics require a {column!r} column.")
    try:
        preferences = pd.Series(battles[column], dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Pairwise preferences must be numeric.") from exc
    parsed = preferences.dropna()
    invalid = ~np.isfinite(parsed) | ~parsed.between(0, 1)
    if invalid.any():
        raise ValueError(
            "Pairwise preferences must be missing or finite values in [0, 1]."
        )
    return preferences


def _pairwise_view(battles: pd.DataFrame) -> pd.DataFrame:
    """Normalize rows carrying an evaluation model to the focal/opponent view."""
    if "evaluation_model" not in battles:
        return battles
    frame = battles.loc[battles["evaluation_model"].notna()].copy()
    required = {"model_a", "model_b", "pref"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Evaluation battles are missing columns: {missing}.")
    focal_is_a = frame["model_a"] == frame["evaluation_model"]
    focal_is_b = frame["model_b"] == frame["evaluation_model"]
    if not (focal_is_a ^ focal_is_b).all():
        raise ValueError(
            "Each evaluation battle must contain its evaluation model exactly once."
        )
    frame["model"] = frame["evaluation_model"]
    frame["baseline"] = frame["model_b"].where(focal_is_a, frame["model_a"])
    frame["pref"] = frame["pref"].where(focal_is_a, 1 - frame["pref"])
    if {"completion_a", "completion_b"} <= set(frame.columns):
        frame["completion_model"] = frame["completion_a"].where(
            focal_is_a, frame["completion_b"]
        )
        frame["completion_baseline"] = frame["completion_b"].where(
            focal_is_a, frame["completion_a"]
        )
    return frame


def _format_winrate(value: object) -> str:
    if value is None or pd.isna(value):
        return "unavailable"
    return f"{float(value):.2%}"


@dataclass(frozen=True, kw_only=True)
class PairwiseWinRateMetric:
    """Configured candidate win-rate calculation."""

    def calculate(self, battles: pd.DataFrame) -> dict[str, object]:
        frame = _pairwise_view(battles)
        preferences = _preferences(frame)
        return compute_pref_summary(preferences).to_dict()

    @staticmethod
    def render(result: dict[str, object]) -> str:
        return (
            f"pairwise_win_rate: {_format_winrate(result['winrate'])} "
            f"({result['num_wins']} wins, {result['num_losses']} losses, "
            f"{result['num_ties']} ties, {result['num_missing']} missing)"
        )


def collapse_pairwise_battles(battles: pd.DataFrame) -> pd.DataFrame:
    """Collapse answer-order judgments into one row per physical battle."""
    frame = _pairwise_view(battles).copy()
    required = {
        "instruction_index",
        "model",
        "baseline",
        "completion_model",
        "completion_baseline",
        "pref",
        "orientation",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Length control requires battle columns: {missing}.")

    frame["pref"] = _preferences(frame)
    if frame.empty:
        frame["n_judgments"] = pd.Series(dtype="int64")
        frame["n_parsed"] = pd.Series(dtype="int64")
        return frame
    keys = ["instruction_index", "model", "baseline"]
    if "category" in frame:
        keys.append("category")
    grouped = frame.groupby(keys, sort=False, dropna=False)
    if any(group["orientation"].duplicated().any() for _, group in grouped):
        raise ValueError("A physical battle contains duplicate orientations.")

    orientations = set(frame["orientation"])
    allowed = {"single", "direct", "reversed"}
    if not orientations <= allowed:
        raise ValueError(
            f"Unknown battle orientations: {sorted(orientations - allowed)}."
        )
    if orientations == {"single"}:
        expected = {"single"}
    elif orientations == {"direct", "reversed"}:
        expected = {"direct", "reversed"}
    else:
        raise ValueError(
            "Battles require one single orientation or complete direct/reversed pairs."
        )
    rows: list[dict[str, object]] = []
    for _, group in frame.groupby(keys, sort=False, dropna=False):
        group_orientations = group["orientation"].tolist()
        if set(group_orientations) != expected:
            raise ValueError(
                "Every physical battle must contain the expected orientations."
            )
        for column in ("completion_model", "completion_baseline"):
            values = group[column].tolist()
            if any(not isinstance(value, str) for value in values):
                raise ValueError("Length control requires string completions.")
            if len(set(values)) != 1:
                raise ValueError(
                    "A physical battle has different completions across orientations."
                )

        parsed = group["pref"].dropna()
        row = group.iloc[0].to_dict()
        row["pref"] = float(parsed.mean()) if len(parsed) else float("nan")
        row["n_judgments"] = len(group)
        row["n_parsed"] = len(parsed)
        rows.append(row)

    return pd.DataFrame(rows)


def _has_separation(x: np.ndarray, outcomes: np.ndarray) -> bool:
    positive = x[outcomes > 0]
    negative = x[outcomes < 1]
    if not len(positive) or not len(negative):
        return True
    return bool(positive.max() <= negative.min() or negative.max() <= positive.min())


def _fit_length_model(
    length_difference: np.ndarray, outcomes: np.ndarray
) -> tuple[float, float]:
    scale = float(np.std(length_difference, ddof=1))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Response length differences have no sample variance.")
    x = np.asarray(length_difference / scale, dtype="float64")
    y = np.asarray(outcomes, dtype="float64")
    if _has_separation(x, y):
        raise ValueError("Length-controlled logistic regression is separated.")

    design = np.repeat(x.reshape(-1, 1), 2, axis=0)
    labels = np.tile([0, 1], len(y))
    weights = np.column_stack((1 - y, y)).ravel()
    model = LogisticRegression(
        fit_intercept=True, C=np.inf, solver="lbfgs", max_iter=1000, tol=1e-10
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(design, labels, sample_weight=weights)
    intercept = float(model.intercept_[0])
    coefficient = float(model.coef_[0, 0])
    if not math.isfinite(intercept) or not math.isfinite(coefficient):
        raise ValueError("Length-controlled logistic regression did not converge.")
    return intercept, coefficient


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(min(value, 709), -709)))


def _bootstrap_interval(
    length_difference: np.ndarray, outcomes: np.ndarray
) -> list[float] | None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scores: list[float] = []
    for _ in range(BOOTSTRAP_ROUNDS):
        sample = rng.integers(0, len(outcomes), size=len(outcomes))
        try:
            intercept, _ = _fit_length_model(
                length_difference[sample], outcomes[sample]
            )
        except (ValueError, ConvergenceWarning):
            continue
        scores.append(_sigmoid(intercept))
    if len(scores) < 0.95 * BOOTSTRAP_ROUNDS:
        return None
    alpha = (1 - CONFIDENCE_LEVEL) / 2
    low, high = np.quantile(scores, [alpha, 1 - alpha])
    return [float(low), float(high)]


@dataclass(frozen=True, kw_only=True)
class LengthControlledWinrateMetric:
    """Configured equal-length candidate win-rate calculation."""

    def calculate(self, battles: pd.DataFrame) -> dict[str, object]:
        collapsed = collapse_pairwise_battles(battles)
        result: dict[str, object] = {
            "num_pairs": len(collapsed),
            "num_scored": int(
                (
                    collapsed["pref"].notna()
                    & (collapsed["n_parsed"] == collapsed["n_judgments"])
                ).sum()
            ),
        }
        if collapsed.empty:
            return {**result, "winrate": None}
        if collapsed["model"].nunique(dropna=False) != 1:
            raise ValueError("Length control requires exactly one evaluation model.")
        if collapsed["baseline"].nunique(dropna=False) != 1:
            raise ValueError("Length control requires exactly one baseline model.")

        complete = collapsed.loc[
            collapsed["pref"].notna()
            & (collapsed["n_parsed"] == collapsed["n_judgments"])
        ]
        if len(complete) < 3:
            return {**result, "winrate": None}

        length_difference = np.array(
            [
                len(candidate) - len(baseline)
                for candidate, baseline in zip(
                    complete["completion_model"],
                    complete["completion_baseline"],
                    strict=True,
                )
            ],
            dtype="float64",
        )
        outcomes = 1 - complete["pref"].to_numpy(dtype="float64")
        result["equal_length_extrapolation"] = not (
            length_difference.min() <= 0 <= length_difference.max()
        )
        try:
            intercept, _ = _fit_length_model(length_difference, outcomes)
        except (ValueError, ConvergenceWarning):
            return {**result, "winrate": None}

        result.update(
            {
                "winrate": _sigmoid(intercept),
                "confidence_interval": _bootstrap_interval(length_difference, outcomes),
            }
        )
        return result

    @staticmethod
    def render(result: dict[str, object]) -> str:
        line = (
            "length_controlled_winrate: "
            f"{_format_winrate(result['winrate'])} "
            f"({result['num_scored']}/{result['num_pairs']} scored pairs)"
        )
        interval = result.get("confidence_interval")
        if interval is not None:
            line += f" [{interval[0]:.2%}, {interval[1]:.2%}]"
        return line
