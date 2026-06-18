"""Cohen's kappa between judge and human hard win/loss labels."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def language_kappa(judge_hard: pd.Series, human_hard: pd.Series) -> float:
    """Cohen's kappa over hard win/loss labels (0.0=A wins, 1.0=B wins).

    Ties (0.5) and NaN on either side are excluded. Returns NaN when fewer
    than two comparable battles remain or either side is single-class (kappa
    is undefined there).
    """
    j = pd.Series(judge_hard, dtype="float64").reset_index(drop=True)
    h = pd.Series(human_hard, dtype="float64").reset_index(drop=True)
    mask = (j != 0.5) & (h != 0.5) & j.notna() & h.notna()
    j, h = j[mask], h[mask]
    if len(j) < 2 or j.nunique() < 2 or h.nunique() < 2:
        return float("nan")
    return float(cohen_kappa_score(h, j))
