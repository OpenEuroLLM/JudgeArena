"""Official Arena-Hard-Auto pairwise scoring (arena-hard-v0.1 show_result.py).

Decisive verdicts count as three battles, and the reported score is the win
fraction of the weighted battles. With one model against one baseline the
leaderboard's Bradley-Terry fit reduces exactly to this fraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from judgearena.utils.eval import PrefSummary

DECISIVE_WEIGHT = 3
BOOTSTRAP_ROUNDS = 100


def _outcomes(prefs: pd.Series) -> np.ndarray:
    """Expand graded preferences into weighted battle outcomes for model A."""
    outcomes: list[float] = []
    for pref in pd.Series(prefs, dtype="float64").dropna():
        outcome = 1.0 if pref < 0.5 else 0.0 if pref > 0.5 else 0.5
        weight = DECISIVE_WEIGHT if pref in (0.0, 1.0) else 1
        outcomes.extend([outcome] * weight)
    return np.asarray(outcomes)


def score(battles: pd.DataFrame):
    """Return the official weighted summary and bootstrap interval."""
    # Imported lazily to avoid a module cycle with the scorer registry.
    from judgearena.benchmarks.pairwise.scoring import ScoringResult

    raw = battles["pref"]
    outcomes = _outcomes(raw)
    num_missing = int(raw.isna().sum())
    summary = PrefSummary(
        num_battles=len(outcomes) + num_missing,
        winrate=float(outcomes.mean()) if len(outcomes) else float("nan"),
        num_wins=int((outcomes == 1.0).sum()),
        num_losses=int((outcomes == 0.0).sum()),
        num_ties=int((outcomes == 0.5).sum()),
        num_missing=num_missing,
    )

    metrics: dict[str, float | None] = {
        "score_ci_low": None,
        "score_ci_high": None,
    }
    if len(outcomes):
        rng = np.random.default_rng(0)
        samples = rng.integers(0, len(outcomes), size=(BOOTSTRAP_ROUNDS, len(outcomes)))
        scores = outcomes[samples].mean(axis=1)
        metrics["score_ci_low"] = float(np.percentile(scores, 2.5))
        metrics["score_ci_high"] = float(np.percentile(scores, 97.5))

    return ScoringResult(
        summary=summary,
        metrics=metrics,
        scoring_details={
            "decisive_weight": DECISIVE_WEIGHT,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        },
    )
