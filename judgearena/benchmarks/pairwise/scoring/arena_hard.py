"""Official Arena-Hard-Auto pairwise scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from judgearena.utils.eval import PrefSummary

DECISIVE_WEIGHT = 3
BOOTSTRAP_ROUNDS = 100
CI_LOWER_QUANTILE = 0.05
CI_UPPER_QUANTILE = 0.95
CONFIDENCE_LEVEL = 0.90
V20_BOOTSTRAP_SEED = 0
V01_CI_LOWER_QUANTILE = 0.025
V01_CI_UPPER_QUANTILE = 0.975
V01_CONFIDENCE_LEVEL = 0.95
STYLE_CONTROLLED_CATEGORIES = frozenset({"hard_prompt"})
STYLE_FEATURES = ("length", "headers", "lists", "bold")
MODEL_STYLE_COLUMNS = tuple(f"model_style_{index}" for index in range(4))
BASELINE_STYLE_COLUMNS = tuple(f"baseline_style_{index}" for index in range(4))
_OFFICIAL_JUDGE_CONFIG = {
    "gpt-4.1": {
        "prompt_preset": "arena-hard",
        "temperature": 0.0,
        "max_out_tokens": 16000,
    },
    "gemini-2.5": {
        "prompt_preset": "arena-hard",
        "temperature": 1.0,
        "max_out_tokens": 32000,
    },
}
_CODE_BLOCK = re.compile(r"```([^`]*)```")


@lru_cache(maxsize=1)
def _load_style_calibration() -> pd.DataFrame:
    """Load the compact joint population derived from pinned upstream data."""
    resource = resources.files(__package__).joinpath(
        "arena_hard_v20_calibration.csv.gz"
    )
    with resource.open("rb") as calibration_file:
        return pd.read_csv(calibration_file, compression="gzip")


def _fit_model_id(model: object) -> str:
    """Match upstream's path removal while preserving model-id case."""
    return str(model).rsplit("/", 1)[-1]


def _calibration_judge(judge: object) -> str:
    requested = _fit_model_id(judge).lower()
    supported = {name.lower(): name for name in _OFFICIAL_JUDGE_CONFIG}
    if requested not in supported:
        raise ValueError(
            "Official Arena-Hard v2 style-controlled scoring has no calibration "
            f"for judge {judge!r}; available judges: {sorted(supported.values())}."
        )
    return supported[requested]


def _validate_calibration_protocol(
    *,
    judge: object,
    prompt_preset: object,
    temperature: object,
    max_out_tokens: object,
    swap_mode: object,
) -> str:
    judge_id = _calibration_judge(judge)
    protocol = _OFFICIAL_JUDGE_CONFIG[judge_id]
    actual = {
        "judge_prompt_preset": prompt_preset,
        "judge_temperature": temperature,
        "judge_max_out_tokens": max_out_tokens,
        "judge_swap_mode": swap_mode,
    }
    expected = {
        "judge_prompt_preset": protocol["prompt_preset"],
        "judge_temperature": protocol["temperature"],
        "judge_max_out_tokens": protocol["max_out_tokens"],
        "judge_swap_mode": "both",
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            raise ValueError(
                f"Arena-Hard v2 calibration for judge {judge_id!r} requires "
                f"{field}={expected_value!r}; found {actual[field]!r}."
            )
    return judge_id


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
    valid, _ = _official_battles(battles, require_complete_pairs=True)
    prefs = valid["pref"].astype("float64")
    weights = np.where(prefs.isin((0.0, 1.0)), DECISIVE_WEIGHT, 1)
    expanded = valid.iloc[np.repeat(np.arange(len(valid)), weights)].copy()
    expanded["outcome"] = np.where(
        expanded["pref"] < 0.5,
        1.0,
        np.where(expanded["pref"] > 0.5, 0.0, 0.5),
    )
    return expanded.reset_index(drop=True)


def _official_battles(
    battles: pd.DataFrame, *, require_complete_pairs: bool
) -> tuple[pd.DataFrame, int]:
    """Select parseable rows under one Arena-Hard protocol."""
    frame = battles.copy()
    if "pref" not in frame:
        raise ValueError("Arena-Hard metrics require a 'pref' column.")
    try:
        frame["pref"] = pd.Series(frame["pref"], dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("Arena-Hard preferences must be numeric.") from exc
    parsed = frame["pref"].dropna()
    if (~np.isfinite(parsed) | ~parsed.between(0, 1)).any():
        raise ValueError(
            "Arena-Hard preferences must be missing or finite values in [0, 1]."
        )
    valid = frame["pref"].notna()
    if (
        not require_complete_pairs
        or "orientation" not in frame
        or "reversed" not in set(frame["orientation"])
    ):
        return frame.loc[valid], int((~valid).sum())

    pair_columns = [
        column
        for column in (
            "instruction_index",
            "model",
            "baseline",
            "category",
            "judge",
        )
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


def _summarize_battles(
    battles: pd.DataFrame, *, require_complete_pairs: bool
) -> PrefSummary:
    valid, num_missing = _official_battles(
        battles, require_complete_pairs=require_complete_pairs
    )
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
    require_complete_pairs: bool,
) -> tuple[float | None, float | None, float | None]:
    valid, _ = _official_battles(battles, require_complete_pairs=require_complete_pairs)
    outcomes = _outcomes(valid["pref"])
    if not len(outcomes):
        return None, None, None
    # RandomState reproduces upstream v0.1's np.random.seed(...), followed by
    # repeated DataFrame.sample(..., replace=True), without mutating global RNG.
    rng = np.random.RandomState(seed)
    samples = rng.randint(0, len(outcomes), size=(BOOTSTRAP_ROUNDS, len(outcomes)))
    scores = outcomes[samples].mean(axis=1)
    return (
        float(scores.mean()),
        float(np.percentile(scores, lower_quantile * 100)),
        float(np.percentile(scores, upper_quantile * 100)),
    )


def _add_live_style_features(battles: pd.DataFrame) -> pd.DataFrame:
    """Compute the four Arena-Hard style features for live completions."""
    frame = battles.copy()
    for columns, completion_column in (
        (MODEL_STYLE_COLUMNS, "completion_model"),
        (BASELINE_STYLE_COLUMNS, "completion_baseline"),
    ):
        if set(columns).issubset(frame) and frame[list(columns)].notna().all().all():
            continue
        if completion_column not in frame or frame[completion_column].isna().any():
            raise ValueError(f"Arena-Hard v2 battles require {completion_column!r}.")
        frame[list(columns)] = np.vstack(frame[completion_column].map(_style_features))
    return frame


def _single_value(frame: pd.DataFrame, column: str) -> object:
    if column not in frame:
        raise ValueError(f"Arena-Hard v2 battles require {column!r}.")
    values = frame[column].dropna().unique().tolist()
    if len(values) != 1:
        raise ValueError(f"Arena-Hard v2 requires one {column}; found {values}.")
    return values[0]


def _select_calibration(battles: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Select the judge population and replace any copy of the live candidate."""
    judge = _single_value(battles, "judge")
    candidate = _single_value(battles, "model")
    judge_id = _validate_calibration_protocol(
        judge=judge,
        prompt_preset=_single_value(battles, "judge_prompt_preset"),
        temperature=_single_value(battles, "judge_temperature"),
        max_out_tokens=_single_value(battles, "judge_max_out_tokens"),
        swap_mode="both" if "reversed" in set(battles["orientation"]) else "fixed",
    )

    calibration = _load_style_calibration()
    categories = set(battles["category"].dropna())
    population = calibration.loc[
        (calibration["judge"] == judge_id) & calibration["category"].isin(categories)
    ]
    selected = population.loc[
        population["model"].map(_fit_model_id) != _fit_model_id(candidate)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"Arena-Hard v2 has no calibration models for judge {judge_id!r}."
        )

    expected_ids = set(population["instruction_index"])
    live_ids = set(battles["instruction_index"])
    unknown_ids = sorted(live_ids - expected_ids)
    if unknown_ids:
        raise ValueError(
            f"Arena-Hard v2 battles contain unknown instruction IDs: {unknown_ids[:5]}."
        )
    return selected, live_ids == expected_ids


def _build_joint_style_design(
    battles: pd.DataFrame,
    calibration_battles: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build upstream's joint model and normalized style-control features."""
    live_weighted = _add_live_style_features(_weighted_battles(battles))
    calibration_weighted = _weighted_battles(calibration_battles)
    weighted = pd.concat(
        [calibration_weighted, live_weighted], ignore_index=True, sort=False
    )

    baselines = weighted["baseline"].dropna().unique().tolist()
    if len(baselines) != 1:
        raise ValueError(
            "Arena-Hard style-controlled categories require one shared baseline; "
            f"found {baselines}."
        )
    baseline = _fit_model_id(baselines[0])
    fit_models = weighted["model"].map(_fit_model_id)
    fit_baselines = weighted["baseline"].map(_fit_model_id)
    models = sorted(set(fit_models) | {baseline})
    model_index = {model: index for index, model in enumerate(models)}

    model_features = np.zeros((len(weighted), len(models)), dtype="float32")
    for row, (model, row_baseline) in enumerate(
        zip(fit_models, fit_baselines, strict=True)
    ):
        model_features[row, model_index[model]] = 1.0
        model_features[row, model_index[row_baseline]] -= 1.0

    candidate_style = weighted.loc[:, MODEL_STYLE_COLUMNS].to_numpy(dtype="float32")
    baseline_style = weighted.loc[:, BASELINE_STYLE_COLUMNS].to_numpy(dtype="float32")
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
    if not np.isfinite(style_difference).all() or not np.isfinite(scale).all():
        raise ValueError("Arena-Hard joint style features must be finite.")
    if (scale == 0).any():
        raise ValueError("Arena-Hard joint style features must have nonzero variance.")
    normalized_style = centered / scale
    return (
        np.column_stack((model_features, normalized_style)),
        weighted["outcome"].to_numpy(dtype="float32"),
        models,
    )


def _logistic_coefficients(features: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Return joint Bradley-Terry and style coefficients from scikit-learn."""
    # Sample weights express a tie as half a loss and half a win.
    design = np.repeat(features, 2, axis=0)
    labels = np.tile([0.0, 1.0], len(outcomes))
    weights = np.column_stack((1.0 - outcomes, outcomes)).ravel()
    model = LogisticRegression(
        fit_intercept=False,
        C=np.inf,
        solver="lbfgs",
        tol=1e-9,
        max_iter=1000,
    ).fit(design, labels, sample_weight=weights)
    return model.coef_[0].astype("float32")


def _bootstrap_style_scores(
    battles: pd.DataFrame,
    calibration_battles: pd.DataFrame,
) -> dict[str, tuple[float, float, float]]:
    """Return median and 5th/95th percentile scores for each candidate model."""
    features, outcomes, models = _build_joint_style_design(battles, calibration_battles)
    baseline = _fit_model_id(battles["baseline"].iloc[0])
    baseline_index = models.index(baseline)
    rng = np.random.RandomState(V20_BOOTSTRAP_SEED)
    coefficients = np.vstack(
        [
            _logistic_coefficients(features[indices], outcomes[indices])
            for indices in rng.randint(
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
    """Return Arena-Hard v2.0's category-scoped results."""
    summaries: dict[str, dict[str, object]] = {}
    for category, category_battles in battles.groupby("category", sort=False):
        baseline = _single_value(category_battles, "baseline")
        summary = _summarize_battles(
            category_battles, require_complete_pairs=True
        ).to_dict()
        scoring_method = "weighted_mean"

        if category in STYLE_CONTROLLED_CATEGORIES:
            model = _single_value(category_battles, "model")
            summary["raw_winrate"] = summary["winrate"]
            scoring_method = "joint_style_controlled_bt"
            valid, _ = _official_battles(category_battles, require_complete_pairs=True)
            if valid.empty:
                ci_low = ci_high = None
                summary["official_population_complete"] = False
            else:
                calibration, complete = _select_calibration(category_battles)
                score, ci_low, ci_high = _bootstrap_style_scores(
                    category_battles,
                    calibration,
                )[_fit_model_id(model)]
                summary["winrate"] = score
                summary["official_population_complete"] = complete
        else:
            score, ci_low, ci_high = _confidence_interval(
                category_battles,
                require_complete_pairs=True,
            )
            if score is not None:
                summary["winrate"] = score

        summaries[str(category)] = {
            **summary,
            "baseline_model": baseline,
            "score_ci_low": ci_low,
            "score_ci_high": ci_high,
            "scoring_method": scoring_method,
        }
    return summaries


@dataclass(frozen=True, kw_only=True)
class ArenaHardV01Metric:
    """Official Arena-Hard v0.1 weighted score and uncertainty."""

    def calculate(self, battles: pd.DataFrame) -> dict[str, object]:
        _, ci_low, ci_high = _confidence_interval(
            battles,
            lower_quantile=V01_CI_LOWER_QUANTILE,
            upper_quantile=V01_CI_UPPER_QUANTILE,
            seed=42,
            require_complete_pairs=False,
        )
        return {
            **_summarize_battles(battles, require_complete_pairs=False).to_dict(),
            "score_ci_low": ci_low,
            "score_ci_high": ci_high,
            "decisive_weight": DECISIVE_WEIGHT,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "confidence_level": V01_CONFIDENCE_LEVEL,
            "confidence_quantiles": [
                V01_CI_LOWER_QUANTILE,
                V01_CI_UPPER_QUANTILE,
            ],
            "official_scope": "overall",
        }

    @staticmethod
    def render(result: dict[str, object]) -> str:
        line = f"arena_hard_v01: {result['winrate']:.2%}"
        if result["score_ci_low"] is not None:
            line += f" [{result['score_ci_low']:.2%}, {result['score_ci_high']:.2%}]"
        return line


@dataclass(frozen=True, kw_only=True)
class ArenaHardV20Metric:
    """Official Arena-Hard v2 category-scoped score."""

    def calculate(self, battles: pd.DataFrame) -> dict[str, object]:
        result: dict[str, object] = {
            **_summarize_battles(battles, require_complete_pairs=True).to_dict(),
            "decisive_weight": DECISIVE_WEIGHT,
            "bootstrap_rounds": BOOTSTRAP_ROUNDS,
            "bootstrap_seed": V20_BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "confidence_quantiles": [CI_LOWER_QUANTILE, CI_UPPER_QUANTILE],
            "official_scope": "per_category",
            "aggregate_score_is_official": False,
            "style_control_features": list(STYLE_FEATURES),
        }
        categories = battles["category"].dropna().unique().tolist()
        result["category_methods"] = {
            str(category): (
                "joint_style_controlled_bt"
                if category in STYLE_CONTROLLED_CATEGORIES
                else "weighted_mean"
            )
            for category in categories
        }
        if len(categories) == 1:
            result.update(_summarize_by_category(battles)[str(categories[0])])
        return result

    @staticmethod
    def render(result: dict[str, object]) -> str:
        line = f"arena_hard_v20: {result['winrate']:.2%}"
        if result.get("score_ci_low") is not None:
            line += f" [{result['score_ci_low']:.2%}, {result['score_ci_high']:.2%}]"
        if result.get("scoring_method"):
            line += f" ({result['scoring_method']})"
        elif result.get("aggregate_score_is_official") is False:
            line += " (unofficial aggregate; see category scores)"
        return line
