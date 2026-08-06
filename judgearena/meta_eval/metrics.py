"""Metrics for judge meta-evaluation against human labels."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from judgearena.estimate_elo_ratings import fit_bradley_terry

WINNER_LABELS = ["model_a", "model_b", "tie"]


def _cohen_kappa(y_true: list[str], y_pred: list[str]) -> float:
    if len(set(y_true) | set(y_pred)) < 2:
        return float("nan")
    return float(cohen_kappa_score(y_true, y_pred, labels=WINNER_LABELS))


def _finite_std(values: list[float]) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    return float(np.std(finite)) if len(finite) else float("nan")


def bootstrap_std(
    y_true: list[str],
    y_pred: list[str],
    *,
    n_bootstraps: int,
    seed: int,
) -> tuple[float, float]:
    if not y_true:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    acc_samples: list[float] = []
    kappa_samples: list[float] = []
    for _ in range(n_bootstraps):
        idx = rng.choice(len(y_true_arr), size=len(y_true_arr), replace=True)
        acc_samples.append(float(np.mean(y_true_arr[idx] == y_pred_arr[idx])))
        kappa_samples.append(
            _cohen_kappa(
                y_true_arr[idx].tolist(),
                y_pred_arr[idx].tolist(),
            )
        )
    return float(np.std(acc_samples)), _finite_std(kappa_samples)


def compute_agreement_metrics(
    winner_human: list[str],
    winner_llm: list[str],
    *,
    n_bootstraps: int,
    seed: int,
) -> dict[str, float | int]:
    n_all = len(winner_human)
    if n_all == 0:
        nan = float("nan")
        return {
            "n": 0,
            "accuracy": nan,
            "acc_se": nan,
            "kappa": nan,
            "kappa_se": nan,
            "n_nt": 0,
            "accuracy_nt": nan,
            "acc_se_nt": nan,
            "kappa_nt": nan,
            "kappa_se_nt": nan,
        }

    acc_all = (
        sum(h == pred for h, pred in zip(winner_human, winner_llm, strict=True)) / n_all
    )
    kappa_all = _cohen_kappa(winner_human, winner_llm)
    acc_se, kappa_se = bootstrap_std(
        winner_human,
        winner_llm,
        n_bootstraps=n_bootstraps,
        seed=seed,
    )

    no_tie = [
        (h, pred)
        for h, pred in zip(winner_human, winner_llm, strict=True)
        if h != "tie"
    ]
    wh_nt, wl_nt = zip(*no_tie, strict=False) if no_tie else ([], [])
    n_nt = len(wh_nt)
    if n_nt:
        acc_nt = sum(h == pred for h, pred in zip(wh_nt, wl_nt, strict=True)) / n_nt
        kappa_nt = _cohen_kappa(list(wh_nt), list(wl_nt))
        acc_se_nt, kappa_se_nt = bootstrap_std(
            list(wh_nt),
            list(wl_nt),
            n_bootstraps=n_bootstraps,
            seed=seed + 1,
        )
    else:
        acc_nt = float("nan")
        kappa_nt = float("nan")
        acc_se_nt = float("nan")
        kappa_se_nt = float("nan")

    return {
        "n": n_all,
        "accuracy": acc_all,
        "acc_se": acc_se,
        "kappa": kappa_all,
        "kappa_se": kappa_se,
        "n_nt": n_nt,
        "accuracy_nt": acc_nt,
        "acc_se_nt": acc_se_nt,
        "kappa_nt": kappa_nt,
        "kappa_se_nt": kappa_se_nt,
    }


def format_metric(value: float | None, se: float | None, *, digits: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if se is None or not math.isfinite(se):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {se:.{digits}f}"


def _hard_bradley_terry(
    df: pd.DataFrame,
    winner_col: str,
) -> dict[str, float]:
    battles = df[["model_a", "model_b", winner_col]].copy()
    battles["pref"] = battles[winner_col].map(
        {
            "model_a": 0.0,
            "model_b": 1.0,
            "tie": 0.5,
            "tie (bothbad)": 0.5,
        }
    )
    return fit_bradley_terry(battles, pref_col="pref")


def _bt_ratings(df_sub: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    try:
        human = _hard_bradley_terry(
            df_sub[["model_a", "model_b", "winner"]],
            "winner",
        )
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
    human = _hard_bradley_terry(
        df_sub[["model_a", "model_b", "winner"]],
        "winner",
    )
    llm = fit_bradley_terry(
        df_sub[["model_a", "model_b", "pref_llm"]],
        pref_col="pref_llm",
    )
    return human, llm


def _rating_vectors(
    df_sub: pd.DataFrame,
    *,
    soft: bool,
) -> tuple[np.ndarray, np.ndarray]:
    ratings_fn = _bt_ratings_soft if soft else _bt_ratings
    human, llm = ratings_fn(df_sub)
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
    df_sub: pd.DataFrame,
    *,
    n_bootstraps: int,
    seed: int,
    soft: bool = False,
) -> str:
    human, llm = _rating_vectors(df_sub, soft=soft)
    if len(human) == 0:
        return "n/a"
    if len(np.unique(human)) < 2 or len(np.unique(llm)) < 2:
        return "n/a"
    rho, _ = spearmanr(human, llm)
    if rho is None or not math.isfinite(float(rho)):
        return "n/a"
    _, se = _bootstrap_rank_metric(
        human,
        llm,
        metric="spearman",
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    return format_metric(float(rho), se)


def mae_elo_with_se(
    df_sub: pd.DataFrame,
    *,
    n_bootstraps: int,
    seed: int,
    soft: bool = False,
) -> str:
    human, llm = _rating_vectors(df_sub, soft=soft)
    if len(human) == 0:
        return "n/a"
    mae = float(np.mean(np.abs(human - llm)))
    _, se = _bootstrap_rank_metric(
        human,
        llm,
        metric="mae",
        n_bootstraps=n_bootstraps,
        seed=seed,
    )
    return format_metric(mae, se, digits=1)


def summarize_language_splits(
    df_ann: pd.DataFrame,
    *,
    exclude_human_ties: bool,
    n_bootstraps: int,
    seed: int,
) -> dict[str, dict[str, str | int]]:
    rows: dict[str, dict[str, str | int]] = {}
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
                float(metrics["kappa"]),
                float(metrics["kappa_se"]),
            )
            entry["spearman"] = spearman_with_se(
                df_sub,
                n_bootstraps=n_bootstraps,
                seed=seed + 2,
                soft=False,
            )
            entry["spearman_soft"] = spearman_with_se(
                df_sub,
                n_bootstraps=n_bootstraps,
                seed=seed + 3,
                soft=True,
            )
            entry["mae_elo"] = mae_elo_with_se(
                df_sub,
                n_bootstraps=n_bootstraps,
                seed=seed + 4,
                soft=False,
            )
            entry["mae_soft_elo"] = mae_elo_with_se(
                df_sub,
                n_bootstraps=n_bootstraps,
                seed=seed + 5,
                soft=True,
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
) -> pd.DataFrame:
    """Measure ELO error by annotation budget.

    Tie predictions are excluded after sampling so ``num_battles`` remains the
    number of judge annotations purchased, matching the reference experiment.
    """
    df_battles = df_top[["model_a", "model_b", "winner"]].copy()
    human_ratings = _hard_bradley_terry(df_battles, "winner")
    rows: list[dict[str, float | int | str]] = []

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
                model_llm = sample[["model_a", "model_b", "winner_llm"]].rename(
                    columns={"winner_llm": "winner"}
                )
                hybrid = pd.concat([other_human, model_llm], ignore_index=True)
                hybrid_ratings = _hard_bradley_terry(hybrid, "winner")
                if model in hybrid_ratings and model in human_ratings:
                    gaps.append(
                        abs(hybrid_ratings[model] - human_ratings[model]),
                    )
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
