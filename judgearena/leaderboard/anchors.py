"""Curate-time precompute of the constant anchor side of a panel.

These functions run Bradley-Terry over the frozen anchor battles exactly once,
at curation, and cache the results next to the panel so neither submissions nor
the render Space ever refit. The render path (assemble.py) MUST NOT import this.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from judgearena.estimate_elo_ratings import _winner_to_pref, fit_bradley_terry
from judgearena.leaderboard.assemble import _h2h_win
from judgearena.leaderboard.panel import Panel


def _anchor_pref_col(panel: Panel) -> str:
    method = panel.meta.get("scorer", {}).get("method", "soft")
    return "judge_pref_hard" if method == "hard" else "judge_pref"


def _ratings_and_counts(
    battles: pd.DataFrame, pref_col: str, baseline: str | None
) -> tuple[dict[str, float], dict[str, int]]:
    df = pd.DataFrame(
        {
            "model_a": battles["model_a"],
            "model_b": battles["model_b"],
            "pref": battles[pref_col],
        }
    )
    elo = fit_bradley_terry(df, pref_col="pref", baseline_model=baseline)
    counts = pd.concat([battles["model_a"], battles["model_b"]]).value_counts()
    return (
        {m: float(e) for m, e in elo.items()},
        {m: int(counts.get(m, 0)) for m in elo},
    )


def compute_anchor_ratings(panel: Panel) -> dict:
    baseline = panel.meta.get("baseline_model")
    col = _anchor_pref_col(panel)
    battles = panel.battles
    overall: dict[str, float] = {}
    counts_overall: dict[str, int] = {}
    per_lang: dict[str, dict[str, float]] = {}
    counts_per_lang: dict[str, dict[str, int]] = {}
    if battles is not None and len(battles):
        overall, counts_overall = _ratings_and_counts(battles, col, baseline)
        for lang in sorted(battles["lang"].unique()):
            sub = battles[battles["lang"] == lang]
            per_lang[lang], counts_per_lang[lang] = _ratings_and_counts(sub, col, baseline)
    return {
        "overall": overall,
        "per_lang": per_lang,
        "counts_overall": counts_overall,
        "counts_per_lang": counts_per_lang,
    }


def compute_calibration(panel: Panel, *, n_bootstrap: int = 100, seed: int = 0) -> dict:
    """Human-ELO vs Judge-ELO for the anchors, with judge-side bootstrap CIs."""
    battles = panel.battles
    if battles is None or len(battles) == 0:
        return {"mae": float("nan"), "spearman": float("nan"), "points": []}

    baseline = panel.meta.get("baseline_model")
    judge_col = _anchor_pref_col(panel)

    human_df = pd.DataFrame(
        {
            "model_a": battles["model_a"],
            "model_b": battles["model_b"],
            "pref_hard": battles["human_winner"].map(_winner_to_pref),
        }
    )
    judge_df = pd.DataFrame(
        {
            "model_a": battles["model_a"],
            "model_b": battles["model_b"],
            "pref": battles[judge_col],
        }
    )
    human_elo = fit_bradley_terry(human_df, pref_col="pref_hard", baseline_model=baseline)
    judge_elo = fit_bradley_terry(judge_df, pref_col="pref", baseline_model=baseline)

    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {}
    for _ in range(n_bootstrap):
        sample = judge_df.sample(
            n=len(judge_df), replace=True, random_state=int(rng.integers(0, 2**31))
        )
        ratings = fit_bradley_terry(sample, pref_col="pref", baseline_model=baseline)
        for model, value in ratings.items():
            boot.setdefault(model, []).append(value)

    points = []
    for model in sorted(m for m in human_elo if m in judge_elo):
        vals = boot.get(model, [])
        ci = (
            [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            if vals
            else [float("nan"), float("nan")]
        )
        points.append(
            {
                "model": model,
                "human_elo": float(human_elo[model]),
                "judge_elo": float(judge_elo[model]),
                "judge_ci": ci,
            }
        )

    mae = (
        float(np.mean([abs(p["human_elo"] - p["judge_elo"]) for p in points]))
        if points
        else float("nan")
    )
    if len(points) >= 2:
        h = pd.Series([p["human_elo"] for p in points])
        j = pd.Series([p["judge_elo"] for p in points])
        spearman = float(h.rank().corr(j.rank()))
    else:
        spearman = float("nan")
    return {"mae": mae, "spearman": spearman, "points": points}


def compute_anchor_h2h(panel: Panel) -> dict:
    wins: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    battles = panel.battles
    if battles is not None and len(battles):
        for _, b in battles.iterrows():
            win_a = _h2h_win(b["judge_pref"])
            if win_a is None:
                continue
            a, bb = str(b["model_a"]), str(b["model_b"])
            wins[(a, bb)] += win_a
            counts[(a, bb)] += 1
            wins[(bb, a)] += 1.0 - win_a
            counts[(bb, a)] += 1
    pairwise: dict[str, dict[str, list]] = {}
    for (r, c), w in wins.items():
        pairwise.setdefault(r, {})[c] = [float(w), int(counts[(r, c)])]
    return {"pairwise": pairwise}


def save_anchor_caches(panel: Panel, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "anchor_ratings.json").write_text(
        json.dumps(compute_anchor_ratings(panel), indent=2) + "\n"
    )
    (directory / "calibration.json").write_text(
        json.dumps(compute_calibration(panel), indent=2) + "\n"
    )
    (directory / "anchor_h2h.json").write_text(
        json.dumps(compute_anchor_h2h(panel), indent=2) + "\n"
    )


def load_anchor_caches(directory: str | Path) -> tuple[dict, dict, dict]:
    directory = Path(directory)
    ratings = json.loads((directory / "anchor_ratings.json").read_text())
    calibration = json.loads((directory / "calibration.json").read_text())
    h2h = json.loads((directory / "anchor_h2h.json").read_text())
    return ratings, calibration, h2h
