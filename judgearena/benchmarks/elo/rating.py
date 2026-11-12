"""Bradley-Terry rating fits and reference scoring rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from judgearena.battles import hard_preferences, has_support


def resample_clusters(clusters, n, rng):
    """Yield row positions for ``n`` bootstrap draws of whole clusters."""
    for _ in range(n):
        yield np.concatenate(
            [clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]
        )


def fit_bradley_terry(
    df: pd.DataFrame,
    pref_col: str = "pref",
    scale: float = 400,
    base: float = 10,
    init_rating: float = 1000,
    baseline_model: str | None = None,
    baseline_rating: float = 1000,
) -> dict[str, float]:
    """Fit Bradley-Terry ratings via weighted logistic regression."""
    df = df.dropna(subset=[pref_col])
    if df.empty:
        return {}

    grouped = (
        df.groupby(["model_a", "model_b", pref_col]).size().reset_index(name="count")
    )
    all_models = sorted(set(grouped["model_a"]) | set(grouped["model_b"]))
    models = pd.Series(np.arange(len(all_models)), index=all_models)

    model_a = grouped["model_a"].map(models).to_numpy()
    model_b = grouped["model_b"].map(models).to_numpy()
    prefs = grouped[pref_col].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n = len(grouped)

    design = np.zeros((2 * n, len(models)))
    top = np.arange(n)
    bottom = n + top
    design[top, model_a] = np.log(base)
    design[top, model_b] = -np.log(base)
    design[bottom, model_a] = np.log(base)
    design[bottom, model_b] = -np.log(base)

    labels = np.concatenate([np.ones(n), np.zeros(n)])
    sample_weights = np.concatenate([(1.0 - prefs) * counts, prefs * counts])
    if sample_weights.sum() == 0:
        return {}

    model = LogisticRegression(fit_intercept=False, C=1e10, tol=1e-6, max_iter=1000)
    model.fit(design, labels, sample_weight=sample_weights)
    ratings = scale * model.coef_[0] + init_rating
    if baseline_model is not None and baseline_model in models.index:
        ratings += baseline_rating - ratings[models[baseline_model]]
    return dict(pd.Series(ratings, index=models.index))


def fit_frozen_rating(
    battles: pd.DataFrame,
    candidate: str,
    references: dict[str, float],
    *,
    pref_col: str = "pref",
    hard: bool = False,
    group_col: str = "split_group",
    n_bootstraps: int = 200,
    seed: int = 42,
) -> dict[str, float | int]:
    """Fit only the candidate, with intervals conditional on frozen reference ratings.

    Preferences use 0 for A wins and 1 for B wins. Resampling whole prompt groups
    keeps repeated judgments and answer orders together. Perfect wins/losses have
    unbounded maximum-likelihood ratings, reported as positive/negative infinity.
    """
    if n_bootstraps < 0:
        raise ValueError("n_bootstraps must be non-negative.")
    if candidate in references:
        raise ValueError("The candidate must not already be a frozen reference model.")
    is_a, is_b = battles.model_a.eq(candidate), battles.model_b.eq(candidate)
    if not (is_a ^ is_b).all():
        raise ValueError("Every comparison must contain the candidate exactly once.")
    opponents = battles.model_b.where(is_a, battles.model_a)
    opponent_ratings = opponents.map(references).to_numpy(dtype=float)
    if not np.isfinite(opponent_ratings).all():
        raise ValueError("Every opponent must have a finite frozen reference rating.")
    preferences = pd.to_numeric(battles[pref_col], errors="raise")
    valid = preferences.notna()
    if not preferences[valid].between(0, 1).all():
        raise ValueError("Preferences must be between zero and one.")
    if not valid.any():
        raise ValueError("No valid candidate judgments remain.")
    if battles.loc[valid, group_col].isna().any():
        raise ValueError("Every valid judgment requires a bootstrap group.")
    if hard:
        preferences = hard_preferences(preferences)
    targets = preferences.where(is_b, 1 - preferences)[valid].to_numpy(dtype=float)
    opponents = opponent_ratings[valid.to_numpy()]
    clusters = list(battles.loc[valid].groupby(group_col, sort=False).indices.values())

    def fit(positions):
        target = targets[positions].mean()
        if target == 0 or target == 1:
            return float("-inf") if target == 0 else float("inf")
        ratings = opponents[positions]
        shift = 400 * np.log10(target / (1 - target))
        lower, upper = ratings.min() + shift, ratings.max() + shift
        if lower == upper:
            return float(lower)
        return float(
            brentq(
                lambda rating: (
                    expit((rating - ratings) * np.log(10) / 400).mean() - target
                ),
                lower - 1,
                upper + 1,
            )
        )

    rng = np.random.default_rng(seed)
    samples = [
        fit(positions) for positions in resample_clusters(clusters, n_bootstraps, rng)
    ]
    # Order-statistic quantiles preserve infinite endpoints without inf-inf arithmetic.
    low, high = (
        np.quantile(samples, [0.025, 0.975], method="inverted_cdf")
        if samples
        else (float("nan"), float("nan"))
    )
    return {
        "rating": fit(np.arange(len(targets))),
        "ci_low": float(low),
        "ci_high": float(high),
        "num_comparisons": len(targets),
        "num_missing": int((~valid).sum()),
        "n_bootstraps": n_bootstraps,
    }


def arena_anchor_battles(
    df_arena_all: pd.DataFrame, *, min_battles: int = 500
) -> pd.DataFrame:
    """Build scoring anchors from selected reference comparisons."""
    frame = df_arena_all.loc[:, ["model_a", "model_b", "reference_pref"]].copy()
    frame["pref"] = frame["reference_pref"]
    frame = frame[has_support(frame, min_battles)]
    frame["pref_hard"] = hard_preferences(frame["pref"])
    return frame
