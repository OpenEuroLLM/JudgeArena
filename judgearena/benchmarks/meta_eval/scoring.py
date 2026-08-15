"""Ranking a judge against human Bradley-Terry ratings."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from judgearena.benchmarks.elo.rating import fit_bradley_terry
from judgearena.benchmarks.meta_eval.agreement import (
    compute_agreement_metrics,
    format_metric,
)

LanguageSummary = dict[str, dict[str, str | int]]
RankingFunction = Callable[..., LanguageSummary]


def ranking_annotations(df_ann: pd.DataFrame) -> pd.DataFrame:
    """Keep one stored-order row per sampled battle.

    swap_mode=both writes a forward row and a swapped row; ranking fits must
    not count the same battle twice.
    """
    if "orientation" in df_ann.columns:
        return df_ann[df_ann["orientation"] == "forward"].copy()
    return df_ann.copy()


def _hard_bradley_terry(df: pd.DataFrame, winner_col: str) -> dict[str, float]:
    battles = df[["model_a", "model_b", winner_col]].copy()
    battles["pref"] = battles[winner_col].map(
        {"model_a": 0.0, "model_b": 1.0, "tie": 0.5, "tie (bothbad)": 0.5}
    )
    return fit_bradley_terry(battles, pref_col="pref")


def _bt_ratings(df_sub: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    try:
        human = _hard_bradley_terry(df_sub[["model_a", "model_b", "winner"]], "winner")
        llm = _hard_bradley_terry(
            df_sub[["model_a", "model_b", "winner_llm"]].rename(
                columns={"winner_llm": "winner"}
            ),
            "winner",
        )
    except ValueError:
        return {}, {}
    return human, llm


def _bt_ratings_soft(df_sub: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    human = _hard_bradley_terry(df_sub[["model_a", "model_b", "winner"]], "winner")
    llm = fit_bradley_terry(
        df_sub[["model_a", "model_b", "pref_llm"]], pref_col="pref_llm"
    )
    return human, llm


def _rating_vectors(
    df_sub: pd.DataFrame, *, soft: bool
) -> tuple[np.ndarray, np.ndarray]:
    human, llm = (_bt_ratings_soft if soft else _bt_ratings)(df_sub)
    common = sorted(set(human) & set(llm))
    return (
        np.array([human[model] for model in common]),
        np.array([llm[model] for model in common]),
    )


def _bootstrap_rank_metric(
    hv: np.ndarray,
    lv: np.ndarray,
    *,
    metric: str,
    n_bootstraps: int,
    seed: int,
) -> tuple[float, float]:
    if len(hv) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstraps):
        idx = rng.choice(len(hv), size=len(hv), replace=True)
        if metric == "spearman":
            if len(np.unique(hv[idx])) < 2 or len(np.unique(lv[idx])) < 2:
                continue
            value = float(spearmanr(hv[idx], lv[idx])[0])
        else:
            value = float(np.mean(np.abs(hv[idx] - lv[idx])))
        if math.isfinite(value):
            samples.append(value)
    if not samples:
        return float("nan"), float("nan")
    return float(np.mean(samples)), float(np.std(samples))


def spearman_with_se(
    df_sub: pd.DataFrame, *, n_bootstraps: int, seed: int, soft: bool = False
) -> str:
    human, llm = _rating_vectors(df_sub, soft=soft)
    if len(human) == 0 or len(np.unique(human)) < 2 or len(np.unique(llm)) < 2:
        return "n/a"
    rho, _ = spearmanr(human, llm)
    if rho is None or not math.isfinite(float(rho)):
        return "n/a"
    _, se = _bootstrap_rank_metric(
        human, llm, metric="spearman", n_bootstraps=n_bootstraps, seed=seed
    )
    return format_metric(float(rho), se)


def mae_elo_with_se(
    df_sub: pd.DataFrame, *, n_bootstraps: int, seed: int, soft: bool = False
) -> str:
    human, llm = _rating_vectors(df_sub, soft=soft)
    if len(human) == 0:
        return "n/a"
    mae = float(np.mean(np.abs(human - llm)))
    _, se = _bootstrap_rank_metric(
        human, llm, metric="mae", n_bootstraps=n_bootstraps, seed=seed
    )
    return format_metric(mae, se, digits=1)


def summarize_language_splits(
    df_ann: pd.DataFrame,
    *,
    exclude_human_ties: bool,
    n_bootstraps: int,
    seed: int,
) -> LanguageSummary:
    rows: LanguageSummary = {}
    for label, mask in [
        ("English", df_ann["lang"] == "en"),
        ("Multilingual", df_ann["lang"] != "en"),
    ]:
        df_sub = df_ann[mask]
        if exclude_human_ties:
            df_sub = df_sub[df_sub["winner"] != "tie"]
        metrics = compute_agreement_metrics(
            df_sub["winner"].tolist(),
            df_sub["winner_llm"].tolist(),
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
        entry: dict[str, str | int] = {"n": metrics["n"]}
        if metrics["n"] == 0:
            entry.update(
                {
                    "kappa": "n/a",
                    "spearman": "n/a",
                    "spearman_soft": "n/a",
                    "mae_elo": "n/a",
                    "mae_soft_elo": "n/a",
                }
            )
        else:
            entry["kappa"] = format_metric(
                float(metrics["kappa"]), float(metrics["kappa_se"])
            )
            entry["spearman"] = spearman_with_se(
                df_sub, n_bootstraps=n_bootstraps, seed=seed + 2, soft=False
            )
            entry["spearman_soft"] = spearman_with_se(
                df_sub, n_bootstraps=n_bootstraps, seed=seed + 3, soft=True
            )
            entry["mae_elo"] = mae_elo_with_se(
                df_sub, n_bootstraps=n_bootstraps, seed=seed + 4, soft=False
            )
            entry["mae_soft_elo"] = mae_elo_with_se(
                df_sub, n_bootstraps=n_bootstraps, seed=seed + 5, soft=True
            )
        rows[label] = entry
    return rows


@dataclass(frozen=True)
class MetaEvalScorer:
    """Ranking implementation selected by a meta-eval task's scoring adapter."""

    language_splits: RankingFunction


META_EVAL_SCORERS = {
    "ranking": MetaEvalScorer(language_splits=summarize_language_splits),
}
