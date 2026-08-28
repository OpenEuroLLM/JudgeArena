"""Ranking a judge against human Bradley-Terry ratings."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from judgearena.benchmarks.elo.rating import fit_bradley_terry, winner_to_pref
from judgearena.benchmarks.meta_eval.agreement import (
    agreement_view,
    compute_agreement_metrics,
    format_metric,
)
from judgearena.benchmarks.meta_eval.sampling import require_connected_pool

LanguageSummary = dict[str, dict[str, str | int]]


def ranking_annotations(df_ann: pd.DataFrame) -> pd.DataFrame:
    """Return parsed physical battles shared by agreement and ranking metrics."""
    if "battle_id" in df_ann and df_ann["battle_id"].duplicated().any():
        raise ValueError("Meta-eval scoring requires one row per physical battle.")
    parsed = df_ann.copy()
    if "parse_ok" in parsed:
        parsed = parsed[parsed["parse_ok"]]
    return parsed[parsed["winner_llm"].notna() & parsed["pref_llm"].notna()].copy()


def _fit_connected_preferences(df: pd.DataFrame, *, pref_col: str) -> dict[str, float]:
    battles = df[["model_a", "model_b", pref_col]].dropna().copy()
    if battles.empty:
        raise ValueError("Cannot fit Bradley-Terry ratings without parsed battles.")
    if (battles["model_a"] == battles["model_b"]).any():
        raise ValueError("Bradley-Terry fitting does not allow self-comparisons.")
    models = sorted(set(battles["model_a"]) | set(battles["model_b"]))
    if len(models) < 2:
        raise ValueError("Bradley-Terry fitting requires at least two models.")
    require_connected_pool(battles, models, context="Bradley-Terry fit")
    return fit_bradley_terry(battles, pref_col=pref_col)


def _hard_bradley_terry(df: pd.DataFrame, winner_col: str) -> dict[str, float]:
    battles = df[["model_a", "model_b", winner_col]].copy()
    battles["pref"] = battles[winner_col].map(
        {"model_a": 0.0, "model_b": 1.0, "tie": 0.5}
    )
    if battles["pref"].isna().any():
        invalid = sorted(
            set(battles.loc[battles["pref"].isna(), winner_col].astype(str))
        )
        raise ValueError(
            f"Meta-eval scoring requires canonical winner labels; got {invalid}."
        )
    return _fit_connected_preferences(battles, pref_col="pref")


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
    try:
        human = _hard_bradley_terry(df_sub[["model_a", "model_b", "winner"]], "winner")
        llm = _fit_connected_preferences(
            df_sub[["model_a", "model_b", "pref_llm"]], pref_col="pref_llm"
        )
    except ValueError:
        return {}, {}
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


def compute_elo_gap_summary(
    df_top: pd.DataFrame,
    df_ann: pd.DataFrame,
    top_models: list[str],
    *,
    n_battles_list: list[int],
    n_seeds: int,
    seed: int,
    exclude_ties: bool,
    soft: bool = False,
) -> pd.DataFrame:
    """Measure ELO error by annotation budget.

    Tie predictions are excluded after sampling so ``num_battles`` remains the
    number of judge annotations purchased, matching the reference experiment.
    """
    df_battles = df_top[["model_a", "model_b", "winner"]].copy()
    human_ratings = _hard_bradley_terry(df_battles, "winner")
    rows: list[dict[str, float | int | str | bool]] = []

    for num_battles in n_battles_list:
        for offset in range(n_seeds):
            rng = np.random.default_rng(seed + offset)
            gaps: list[float] = []
            for model in top_models:
                model_mask_ann = (df_ann["model_a"] == model) | (
                    df_ann["model_b"] == model
                )
                if model_mask_ann.sum() < num_battles:
                    continue
                model_mask_top = (df_battles["model_a"] == model) | (
                    df_battles["model_b"] == model
                )
                other_human = df_battles[~model_mask_top].copy()
                sample = df_ann[model_mask_ann].sample(
                    n=num_battles,
                    replace=False,
                    random_state=int(rng.integers(0, 2**32 - 1)),
                )
                if exclude_ties:
                    sample = sample[sample["winner_llm"] != "tie"]
                if sample.empty:
                    continue
                try:
                    if soft:
                        other_human["pref"] = other_human["winner"].map(winner_to_pref)
                        model_llm = sample[["model_a", "model_b", "pref_llm"]].rename(
                            columns={"pref_llm": "pref"}
                        )
                        hybrid = pd.concat(
                            [
                                other_human[["model_a", "model_b", "pref"]],
                                model_llm,
                            ],
                            ignore_index=True,
                        )
                        hybrid_ratings = _fit_connected_preferences(
                            hybrid, pref_col="pref"
                        )
                    else:
                        model_llm = sample[["model_a", "model_b", "winner_llm"]].rename(
                            columns={"winner_llm": "winner"}
                        )
                        hybrid = pd.concat([other_human, model_llm], ignore_index=True)
                        hybrid_ratings = _hard_bradley_terry(hybrid, "winner")
                except ValueError:
                    continue
                if model in hybrid_ratings and model in human_ratings:
                    gaps.append(abs(hybrid_ratings[model] - human_ratings[model]))
            if gaps:
                rows.append(
                    {
                        "num_battles": num_battles,
                        "seed": offset,
                        "mean_gap": float(np.mean(gaps)),
                        "exclude_ties": exclude_ties,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["num_battles", "mean", "se", "exclude_ties"])

    df_rows = pd.DataFrame(rows)
    return (
        df_rows.groupby(["num_battles", "exclude_ties"])["mean_gap"]
        .agg(mean="mean", se=lambda values: values.std() / np.sqrt(len(values)))
        .reset_index()
    )


def score_meta_eval(
    df_top: pd.DataFrame,
    annotations: pd.DataFrame,
    top_models: list[str],
    *,
    n_bootstraps: int,
    include_human_ties: bool,
    elo_gap_battles: list[int],
    elo_gap_seeds: int,
    seed: int,
) -> dict[str, object]:
    """Compute the agreement, ranking, and held-out Elo metrics."""
    ranking = ranking_annotations(annotations)
    metrics = compute_agreement_metrics(
        ranking["winner"].tolist(),
        ranking["winner_llm"].tolist(),
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    agreement = {
        "all": agreement_view(metrics, exclude_human_ties=False),
        "no_human_ties": agreement_view(metrics, exclude_human_ties=True),
    }
    language_summary = summarize_language_splits(
        ranking,
        exclude_human_ties=not include_human_ties,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    elo_gap_kwargs = {
        "df_top": df_top,
        "df_ann": ranking,
        "top_models": top_models,
        "n_battles_list": elo_gap_battles,
        "n_seeds": elo_gap_seeds,
    }
    elo_gap_all = compute_elo_gap_summary(
        **elo_gap_kwargs,
        seed=seed,
        exclude_ties=False,
    )
    elo_gap_exclude_ties = compute_elo_gap_summary(
        **elo_gap_kwargs,
        seed=seed + 1000,
        exclude_ties=True,
    )
    elo_gap_soft = compute_elo_gap_summary(
        **elo_gap_kwargs,
        seed=seed,
        exclude_ties=False,
        soft=True,
    )
    return {
        "agreement": agreement,
        "language_summary": language_summary,
        "elo_gap_all": elo_gap_all.to_dict(orient="records"),
        "elo_gap_exclude_ties": elo_gap_exclude_ties.to_dict(orient="records"),
        "elo_gap_soft": elo_gap_soft.to_dict(orient="records"),
    }


MetaEvalScorer = Callable[..., dict[str, object]]
META_EVAL_SCORERS: dict[str, MetaEvalScorer] = {"ranking": score_meta_eval}


def resolve_meta_eval_scorer(name: str) -> MetaEvalScorer:
    """Return the scorer registered under a task's scoring adapter ID."""
    try:
        return META_EVAL_SCORERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown meta-eval scorer {name!r}; available: {sorted(META_EVAL_SCORERS)}"
        ) from exc
