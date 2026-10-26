"""Pure sampling and Bradley-Terry rating logic for arena-anchored ELO."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from judgearena.battles import hard_preferences


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


def _sample_fingerprint(sampled: pd.DataFrame) -> str:
    rows = [
        {
            "index": int(index) if isinstance(index, int | np.integer) else str(index),
            "question_id": str(row.get("question_id", row.get("instruction_index"))),
            "model_a": str(row["model_a"]),
            "model_b": str(row["model_b"]),
        }
        for index, row in sampled.iterrows()
    ]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def select_seeded_random_arena_battles(
    df_battles: pd.DataFrame,
    *,
    n_battles: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select a shared random battle panel for outside-model ELO estimation."""
    sampled = df_battles.sample(
        n=min(n_battles, len(df_battles)), random_state=seed, replace=False
    )
    metadata: dict[str, object] = {
        "sampling_mode": "seeded_random",
        "random_seed": seed,
        "requested_rows": n_battles,
        "sampled_rows": len(sampled),
        "sampled_original_indices": [
            int(index) if isinstance(index, int | np.integer) else str(index)
            for index in sampled.index
        ],
        "sampled_question_ids": [
            str(value)
            for value in sampled[
                "question_id" if "question_id" in sampled else "instruction_index"
            ].tolist()
        ],
        "sample_fingerprint": _sample_fingerprint(sampled),
    }
    return sampled.reset_index(drop=True), metadata


def arena_anchor_battles(df_arena_all: pd.DataFrame) -> pd.DataFrame:
    """Select human anchors from normalized PreferenceData.human_battles() rows."""
    frame = df_arena_all.loc[:, ["model_a", "model_b", "pref", "source"]].copy()
    counts = pd.concat([frame["model_a"], frame["model_b"]]).value_counts()
    well_represented = set(counts[counts >= 500].index)
    frame = frame[
        frame["model_a"].isin(well_represented)
        & frame["model_b"].isin(well_represented)
    ]
    frame["pref_hard"] = hard_preferences(frame["pref"])
    return frame
