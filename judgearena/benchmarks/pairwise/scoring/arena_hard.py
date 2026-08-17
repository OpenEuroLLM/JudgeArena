"""Official Arena-Hard-Auto pairwise scoring.

Decisive verdicts count as three battles. Arena-Hard v2.0 results are reported
per category because categories may use different baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from judgearena.benchmarks.pairwise.scoring.models import ScoringResult
from judgearena.utils.eval import PrefSummary

DECISIVE_WEIGHT = 3
BOOTSTRAP_ROUNDS = 100
CI_LOWER_QUANTILE = 0.05
CI_UPPER_QUANTILE = 0.95
CONFIDENCE_LEVEL = 0.90


def _outcomes(prefs: pd.Series) -> np.ndarray:
    """Expand graded preferences into weighted battle outcomes for model A."""
    outcomes: list[float] = []
    for pref in pd.Series(prefs, dtype="float64").dropna():
        outcome = 1.0 if pref < 0.5 else 0.0 if pref > 0.5 else 0.5
        weight = DECISIVE_WEIGHT if pref in (0.0, 1.0) else 1
        outcomes.extend([outcome] * weight)
    return np.asarray(outcomes)


def _official_battles(battles: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop unparseable rows, including both orders of an incomplete pair."""
    frame = battles.copy()
    if "orientation" not in frame or "reversed" not in set(frame["orientation"]):
        valid = frame["pref"].notna()
        return frame.loc[valid], int((~valid).sum())

    pair_columns = [
        column
        for column in ("instruction_index", "model", "baseline", "category")
        if column in frame
    ]
    if not pair_columns:
        raise ValueError(
            "Arena-Hard both-order scoring requires an instruction_index column."
        )

    grouped = frame.groupby(pair_columns, sort=False, dropna=False)
    complete = grouped["pref"].transform(
        lambda prefs: len(prefs) == 2 and prefs.notna().all()
    )
    has_both_orders = grouped["orientation"].transform(
        lambda values: set(values) == {"direct", "reversed"}
    )
    keep = complete & has_both_orders
    return frame.loc[keep], int((~keep).sum())


def _summarize_battles(battles: pd.DataFrame) -> PrefSummary:
    valid, num_missing = _official_battles(battles)
    outcomes = _outcomes(valid["pref"])
    return PrefSummary(
        num_battles=len(outcomes) + num_missing,
        winrate=float(outcomes.mean()) if len(outcomes) else float("nan"),
        num_wins=int((outcomes == 1.0).sum()),
        num_losses=int((outcomes == 0.0).sum()),
        num_ties=int((outcomes == 0.5).sum()),
        num_missing=num_missing,
    )


def _confidence_interval(battles: pd.DataFrame) -> tuple[float | None, float | None]:
    valid, _ = _official_battles(battles)
    outcomes = _outcomes(valid["pref"])
    if not len(outcomes):
        return None, None
    rng = np.random.default_rng(0)
    samples = rng.integers(0, len(outcomes), size=(BOOTSTRAP_ROUNDS, len(outcomes)))
    scores = outcomes[samples].mean(axis=1)
    return (
        float(np.percentile(scores, CI_LOWER_QUANTILE * 100)),
        float(np.percentile(scores, CI_UPPER_QUANTILE * 100)),
    )


def _summarize_by_category(battles: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Return Arena-Hard v2.0's official category-scoped results."""
    summaries: dict[str, dict[str, object]] = {}
    for category, category_battles in battles.groupby("category", sort=False):
        baselines = category_battles["baseline"].dropna().unique().tolist()
        if len(baselines) != 1:
            raise ValueError(
                f"Arena-Hard category {category!r} must use exactly one baseline; "
                f"found {baselines}."
            )
        summary = _summarize_battles(category_battles).to_dict()
        ci_low, ci_high = _confidence_interval(category_battles)
        summaries[str(category)] = {
            **summary,
            "baseline_model": baselines[0],
            "score_ci_low": ci_low,
            "score_ci_high": ci_high,
        }
    return summaries


def score(battles: pd.DataFrame) -> ScoringResult:
    """Return official aggregate, grouped results, and scoring details."""
    has_categories = "category" in battles and battles["category"].notna().any()
    metrics: dict[str, float | None] = {}
    grouped_results: dict[str, object] = {}
    scoring_details: dict[str, object] = {
        "decisive_weight": DECISIVE_WEIGHT,
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "confidence_level": CONFIDENCE_LEVEL,
        "confidence_quantiles": [CI_LOWER_QUANTILE, CI_UPPER_QUANTILE],
        "official_scope": "per_category" if has_categories else "overall",
    }
    if has_categories:
        grouped_results["category"] = _summarize_by_category(battles)
        scoring_details["aggregate_score_is_official"] = False
    else:
        metrics["score_ci_low"], metrics["score_ci_high"] = _confidence_interval(
            battles
        )

    return ScoringResult(
        summary=_summarize_battles(battles),
        metrics=metrics,
        grouped_results=grouped_results,
        scoring_details=scoring_details,
    )
