"""Agreement of a judge with human arena labels."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import cohen_kappa_score

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
            _cohen_kappa(y_true_arr[idx].tolist(), y_pred_arr[idx].tolist())
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
    nan = float("nan")
    if n_all == 0:
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
        winner_human, winner_llm, n_bootstraps=n_bootstraps, seed=seed
    )

    no_tie = [
        (h, pred)
        for h, pred in zip(winner_human, winner_llm, strict=True)
        if h != "tie"
    ]
    n_nt = len(no_tie)
    if n_nt:
        wh_nt, wl_nt = zip(*no_tie, strict=True)
        acc_nt = sum(h == pred for h, pred in zip(wh_nt, wl_nt, strict=True)) / n_nt
        kappa_nt = _cohen_kappa(list(wh_nt), list(wl_nt))
        acc_se_nt, kappa_se_nt = bootstrap_std(
            list(wh_nt), list(wl_nt), n_bootstraps=n_bootstraps, seed=seed + 1
        )
    else:
        acc_nt = kappa_nt = acc_se_nt = kappa_se_nt = nan

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


def format_metric(value: float | None, se: float | None, *, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if se is None or not math.isfinite(se):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {se:.{digits}f}"


def agreement_view(
    metrics: dict[str, float | int], *, exclude_human_ties: bool
) -> dict[str, float | int | str]:
    suffix = "_nt" if exclude_human_ties else ""
    n_key = "n_nt" if exclude_human_ties else "n"
    accuracy = float(metrics[f"accuracy{suffix}"])
    accuracy_se = float(metrics[f"acc_se{suffix}"])
    kappa = float(metrics[f"kappa{suffix}"])
    kappa_se = float(metrics[f"kappa_se{suffix}"])
    return {
        "n": int(metrics[n_key]),
        "accuracy": accuracy,
        "accuracy_se": accuracy_se,
        "kappa": kappa,
        "kappa_se": kappa_se,
        "accuracy_formatted": format_metric(accuracy, accuracy_se),
        "kappa_formatted": format_metric(kappa, kappa_se),
    }
