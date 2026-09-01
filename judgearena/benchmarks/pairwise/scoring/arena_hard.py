"""Official Arena-Hard-Auto pairwise scoring."""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from judgearena.benchmarks.pairwise.scoring.models import ScoringResult
from judgearena.utils.eval import PrefSummary

DECISIVE_WEIGHT = 3
BOOTSTRAP_ROUNDS = 100
CI_LOWER_QUANTILE = 0.05
CI_UPPER_QUANTILE = 0.95
CONFIDENCE_LEVEL = 0.90
V01_CI_LOWER_QUANTILE = 0.025
V01_CI_UPPER_QUANTILE = 0.975
V01_CONFIDENCE_LEVEL = 0.95
STYLE_CONTROLLED_CATEGORIES = frozenset({"hard_prompt", "coding", "math"})
STYLE_FEATURES = ("length", "headers", "lists", "bold")
_CODE_BLOCK = re.compile(r"```([^`]*)```")


@lru_cache(maxsize=4096)
def _style_features(completion: str) -> np.ndarray:
    """Return Arena-Hard's token-length and Markdown style features."""
    import tiktoken

    without_code = completion
    for block in _CODE_BLOCK.findall(completion):
        without_code = without_code.replace(block, "")
    return np.asarray(
        [
            len(
                tiktoken.encoding_for_model("gpt-4o").encode(
                    completion, disallowed_special=()
                )
            ),
            sum(
                len(re.findall(rf"^#{{{level}}}\s", without_code, re.MULTILINE))
                for level in range(1, 7)
            ),
            len(re.findall(r"^\s*\d+\.\s", without_code, re.MULTILINE))
            + len(re.findall(r"^\s*[-*+]\s", without_code, re.MULTILINE)),
            len(re.findall(r"\*\*[^*\n]+\*\*", without_code))
            + len(re.findall(r"__[^_\n]+__", without_code)),
        ],
        dtype="float64",
    )


def _outcomes(prefs: pd.Series) -> np.ndarray:
    """Expand graded preferences into weighted battle outcomes for model A."""
    outcomes: list[float] = []
    for pref in pd.Series(prefs, dtype="float64").dropna():
        outcome = 1.0 if pref < 0.5 else 0.0 if pref > 0.5 else 0.5
        weight = DECISIVE_WEIGHT if pref in (0.0, 1.0) else 1
        outcomes.extend([outcome] * weight)
    return np.asarray(outcomes)


def _weighted_battles(battles: pd.DataFrame) -> pd.DataFrame:
    """Expand decisive verdicts while retaining their models and completions."""
    valid, _ = _official_battles(battles)
    prefs = valid["pref"].astype("float64")
    weights = np.where(prefs.isin((0.0, 1.0)), DECISIVE_WEIGHT, 1)
    expanded = valid.iloc[np.repeat(np.arange(len(valid)), weights)].copy()
    expanded["outcome"] = np.where(
        expanded["pref"] < 0.5,
        1.0,
        np.where(expanded["pref"] > 0.5, 0.0, 0.5),
    )
    return expanded.reset_index(drop=True)


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
    prefs = valid["pref"].astype("float64")
    outcomes = _outcomes(prefs)
    return PrefSummary(
        num_battles=len(battles),
        winrate=float(outcomes.mean()) if len(outcomes) else float("nan"),
        num_wins=int((prefs < 0.5).sum()),
        num_losses=int((prefs > 0.5).sum()),
        num_ties=int((prefs == 0.5).sum()),
        num_missing=num_missing,
    )


def _confidence_interval(
    battles: pd.DataFrame,
    *,
    lower_quantile: float = CI_LOWER_QUANTILE,
    upper_quantile: float = CI_UPPER_QUANTILE,
    seed: int = 0,
) -> tuple[float | None, float | None]:
    valid, _ = _official_battles(battles)
    outcomes = _outcomes(valid["pref"])
    if not len(outcomes):
        return None, None
    # RandomState reproduces upstream v0.1's np.random.seed(...), followed by
    # repeated DataFrame.sample(..., replace=True), without mutating global RNG.
    rng = np.random.RandomState(seed)
    samples = rng.randint(0, len(outcomes), size=(BOOTSTRAP_ROUNDS, len(outcomes)))
    scores = outcomes[samples].mean(axis=1)
    return (
        float(np.percentile(scores, lower_quantile * 100)),
        float(np.percentile(scores, upper_quantile * 100)),
    )


def _style_design(battles: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build upstream's model and normalized style-control features."""
    weighted = _weighted_battles(battles)
    baseline = weighted["baseline"].iloc[0]
    models = sorted(set(weighted["model"]) | {baseline})
    model_index = {model: index for index, model in enumerate(models)}

    model_features = np.zeros((len(weighted), len(models)), dtype="float64")
    for row, (model, row_baseline) in enumerate(
        zip(weighted["model"], weighted["baseline"], strict=True)
    ):
        model_features[row, model_index[model]] = 1.0
        model_features[row, model_index[row_baseline]] -= 1.0

    candidate_style = np.vstack(
        [_style_features(text) for text in weighted["completion_model"]]
    )
    baseline_style = np.vstack(
        [_style_features(text) for text in weighted["completion_baseline"]]
    )
    style_difference = np.zeros_like(candidate_style)
    style_difference[:, 0] = (candidate_style[:, 0] - baseline_style[:, 0]) / (
        candidate_style[:, 0] + baseline_style[:, 0]
    )
    candidate_density = candidate_style[:, 1:] / (candidate_style[:, :1] + 1)
    baseline_density = baseline_style[:, 1:] / (baseline_style[:, :1] + 1)
    style_difference[:, 1:] = (candidate_density - baseline_density) / (
        candidate_density + baseline_density + 1
    )

    centered = style_difference - style_difference.mean(axis=0)
    scale = style_difference.std(axis=0, ddof=1)
    normalized_style = np.divide(
        centered,
        scale,
        out=np.zeros_like(centered),
        where=np.isfinite(scale) & (scale != 0),
    )
    return (
        np.column_stack((model_features, normalized_style)),
        weighted["outcome"].to_numpy(dtype="float64"),
        models,
    )


def _fit_bt(features: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Fit the unregularized Bradley--Terry logistic model used upstream."""

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        logits = features @ coefficients
        loss = np.logaddexp(0, logits).sum() - outcomes @ logits
        gradient = features.T @ (expit(logits) - outcomes)
        return float(loss), gradient

    fitted = minimize(
        objective,
        np.full(features.shape[1], 0.5),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 50, "ftol": 1e-9, "gtol": 1e-9},
    )
    if not np.isfinite(fitted.x).all():
        raise ValueError("Arena-Hard style-controlled BT fit did not converge.")
    return fitted.x


def _style_controlled_scores(
    battles: pd.DataFrame,
) -> dict[str, tuple[float, float, float]]:
    """Return median and 5th/95th percentile scores for each candidate model."""
    features, outcomes, models = _style_design(battles)
    baseline = battles["baseline"].iloc[0]
    baseline_index = models.index(baseline)
    rng = np.random.default_rng(0)
    coefficients = np.vstack(
        [
            _fit_bt(features[indices], outcomes[indices])
            for indices in rng.integers(
                0, len(features), size=(BOOTSTRAP_ROUNDS, len(features))
            )
        ]
    )
    model_coefficients = coefficients[:, : len(models)]
    scores = expit(
        model_coefficients - model_coefficients[:, baseline_index, np.newaxis]
    )
    return {
        model: (
            float(np.quantile(scores[:, index], 0.5)),
            float(np.quantile(scores[:, index], CI_LOWER_QUANTILE)),
            float(np.quantile(scores[:, index], CI_UPPER_QUANTILE)),
        )
        for index, model in enumerate(models)
    }


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
        scoring_method = "weighted_mean"
        if category in STYLE_CONTROLLED_CATEGORIES:
            models = category_battles["model"].dropna().unique().tolist()
            if len(models) != 1:
                raise ValueError(
                    f"Arena-Hard run scoring expects one candidate model; found {models}."
                )
            summary["raw_winrate"] = summary["winrate"]
            scoring_method = "style_controlled_bt"
            valid, _ = _official_battles(category_battles)
            if valid.empty:
                ci_low = ci_high = None
            else:
                score, ci_low, ci_high = _style_controlled_scores(category_battles)[
                    models[0]
                ]
                summary["winrate"] = score
        else:
            ci_low, ci_high = _confidence_interval(category_battles)
        summaries[str(category)] = {
            **summary,
            "baseline_model": baselines[0],
            "score_ci_low": ci_low,
            "score_ci_high": ci_high,
            "scoring_method": scoring_method,
        }
    return summaries


def score_v01(battles: pd.DataFrame) -> ScoringResult:
    """Return the upstream v0.1 overall score and 95% bootstrap interval."""
    ci_low, ci_high = _confidence_interval(
        battles,
        lower_quantile=V01_CI_LOWER_QUANTILE,
        upper_quantile=V01_CI_UPPER_QUANTILE,
        seed=42,
    )
    return ScoringResult(
        summary=_summarize_battles(battles),
        metrics={"score_ci_low": ci_low, "score_ci_high": ci_high},
        scoring_details={
            "decisive_weight": DECISIVE_WEIGHT,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "confidence_level": V01_CONFIDENCE_LEVEL,
            "confidence_quantiles": [
                V01_CI_LOWER_QUANTILE,
                V01_CI_UPPER_QUANTILE,
            ],
            "official_scope": "overall",
        },
    )


def score_v20(battles: pd.DataFrame) -> ScoringResult:
    """Return v2 category-scoped scores and scoring details."""
    return ScoringResult(
        summary=_summarize_battles(battles),
        grouped_results={"category": _summarize_by_category(battles)},
        scoring_details={
            "decisive_weight": DECISIVE_WEIGHT,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "confidence_level": CONFIDENCE_LEVEL,
            "confidence_quantiles": [CI_LOWER_QUANTILE, CI_UPPER_QUANTILE],
            "official_scope": "per_category",
            "aggregate_score_is_official": False,
            "category_methods": {
                **{
                    category: "style_controlled_bt"
                    for category in sorted(STYLE_CONTROLLED_CATEGORIES)
                },
                "creative_writing": "weighted_mean",
            },
            "style_control_features": list(STYLE_FEATURES),
        },
    )
