"""Local AlpacaEval 2.0 raw and length-controlled scoring."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn import config_context
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold

from judgearena.benchmarks.pairwise.scoring.models import ScoringResult
from judgearena.log import get_logger
from judgearena.utils.eval import PrefSummary

logger = get_logger(__name__)

_ALPACA_EVAL_REPO = "tatsu-lab/alpaca_eval"
# Match the dataset revision in alpaca-eval-2.0-official.yaml.
_ALPACA_EVAL_REVISION = "2edc6fad8be6b14ea7230aabfd08188da6b8b814"
# Upstream uses lambda=0.2 split over two gamed baselines.
_GAMED_WEIGHT = 0.1


@lru_cache(maxsize=1)
def _load_gamed_data() -> pd.DataFrame:
    """Load AlpacaEval's pinned length-control calibration rows."""
    path = hf_hub_download(
        repo_id=_ALPACA_EVAL_REPO,
        filename="df_gamed.csv",
        repo_type="dataset",
        revision=_ALPACA_EVAL_REVISION,
    )
    return pd.read_csv(path).drop(columns="model")


def _summarize(battles: pd.DataFrame) -> PrefSummary:
    """Return raw win rate over parsed battles."""
    prefs = battles["pref"]
    valid = prefs.dropna()
    return PrefSummary(
        num_battles=len(prefs),
        winrate=float((1 - valid).mean()) if len(valid) else float("nan"),
        num_wins=int((valid < 0.5).sum()),
        num_losses=int((valid > 0.5).sum()),
        num_ties=int((valid == 0.5).sum()),
        num_missing=int(prefs.isna().sum()),
    )


def _official_annotations(battles: pd.DataFrame) -> pd.DataFrame:
    """Map canonical battles onto AlpacaEval's annotation schema."""
    annotations = pd.DataFrame(
        {
            "index": battles["instruction_index"].astype(int),
            "generator_1": battles["baseline"],
            "generator_2": battles["model"],
            "output_1": battles["completion_baseline"],
            "output_2": battles["completion_model"],
            "preference": 2 - battles["pref"],
        }
    )
    return annotations.dropna(subset=["preference"]).reset_index(drop=True)


def _design(frame: pd.DataFrame) -> np.ndarray:
    """Build AlpacaEval's fixed three-column length-control design."""
    return np.column_stack(
        (
            np.tanh(frame["std_delta_len"].to_numpy(dtype=float)),
            frame["instruction_difficulty"].to_numpy(dtype=float),
            frame["not_gamed_baseline"].to_numpy(dtype=float),
        )
    )


def _continuous_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    true_prob: np.ndarray,
    true_sample_weight: np.ndarray,
) -> float:
    """Return weighted log loss for duplicated soft-label observations."""
    target = np.where(y_true == 1, true_prob, 1 - true_prob)
    predicted = np.clip(y_pred, 1e-15, 1 - 1e-15)
    losses = -(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))
    return float(np.mean(losses * true_sample_weight))


def _fit_length_model(
    features: np.ndarray,
    targets: np.ndarray,
    sample_weight: np.ndarray,
) -> LogisticRegressionCV:
    """Fit AlpacaEval's grouped soft-label logistic cross-validation."""
    n_rows = len(targets)
    duplicated_features = np.vstack((features, features))
    labels = np.concatenate((np.ones(n_rows), np.zeros(n_rows)))
    true_prob = np.concatenate((targets, 1 - targets))
    true_sample_weight = np.tile(sample_weight, 2)
    fit_weight = true_prob * true_sample_weight
    groups = np.tile(np.arange(n_rows), 2)

    with config_context(enable_metadata_routing=True):
        scorer = make_scorer(
            _continuous_log_loss,
            response_method="predict_proba",
            greater_is_better=False,
        ).set_score_request(
            true_prob=True,
            true_sample_weight=True,
        )
        model = LogisticRegressionCV(
            cv=GroupKFold(n_splits=5),
            scoring=scorer,
            random_state=123,
            dual=False,
            penalty="l1",
            solver="liblinear",
            n_jobs=None,
            fit_intercept=False,
        ).set_fit_request(sample_weight=True)
        model.fit(
            duplicated_features,
            labels,
            sample_weight=fit_weight,
            groups=groups,
            true_prob=true_prob,
            true_sample_weight=true_sample_weight,
        )
    return model


def _length_controlled_metrics(annotations: pd.DataFrame) -> dict[str, float]:
    """Reproduce AlpacaEval 2.0's length-controlled GLM metrics."""
    models = annotations["generator_2"].unique().tolist()
    baselines = annotations["generator_1"].unique().tolist()
    if len(models) != 1 or len(baselines) != 1:
        raise ValueError("AlpacaEval scoring requires one candidate and one baseline.")

    targets = annotations["preference"].astype(float) - 1
    if not targets.between(0, 1).all():
        raise ValueError(
            "Normalized AlpacaEval preferences must be finite values in [0, 1]."
        )
    raw_winrate = float(targets.mean() * 100)
    if models[0] == baselines[0]:
        predictions = pd.Series(np.full(len(annotations), 0.5))
        return {
            "length_controlled_winrate": 50.0,
            "lc_standard_error": float(predictions.sem() * 100),
            "win_rate": raw_winrate,
        }

    delta_length = annotations["output_1"].str.len() - annotations["output_2"].str.len()
    scale = float(delta_length.std())
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("AlpacaEval length differences must have nonzero variance.")

    gamed = _load_gamed_data()
    difficulty = gamed.drop_duplicates("index").set_index("index")[
        "instruction_difficulty"
    ]
    unknown = sorted(set(annotations["index"]) - set(difficulty.index))
    if unknown:
        raise ValueError(f"Unknown AlpacaEval instruction IDs: {unknown[:5]}.")

    live = pd.DataFrame(
        {
            "preference": targets,
            "std_delta_len": delta_length / scale,
            "instruction_difficulty": annotations["index"].map(difficulty),
            "not_gamed_baseline": True,
        }
    )
    training = pd.concat([gamed, live], ignore_index=True)
    sample_weight = np.where(
        training["not_gamed_baseline"],
        1.0,
        _GAMED_WEIGHT,
    )
    model = _fit_length_model(
        _design(training),
        training["preference"].to_numpy(dtype=float),
        sample_weight,
    )

    zero_length = live.copy()
    zero_length["std_delta_len"] = 0.0
    predicted = model.predict_proba(_design(zero_length))[:, 1]
    return {
        "length_controlled_winrate": float(predicted.mean() * 100),
        "lc_standard_error": float(pd.Series(predicted).sem() * 100),
        "win_rate": raw_winrate,
    }


def score(battles: pd.DataFrame) -> ScoringResult:
    """Return raw and local length-controlled AlpacaEval metrics."""
    if battles["instruction_index"].duplicated().any():
        logger.warning(
            "Battles contain duplicate instructions (swap_mode='both'?); the "
            "official LC protocol judges each instruction once in randomized order."
        )

    upstream_metrics = _length_controlled_metrics(_official_annotations(battles))
    scoring_metrics: dict[str, float] = {}
    for ours, theirs in (
        ("lc_winrate", "length_controlled_winrate"),
        ("lc_standard_error", "lc_standard_error"),
        ("raw_winrate", "win_rate"),
    ):
        value = upstream_metrics.get(theirs)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"Official AlpacaEval scoring did not return a finite {theirs!r}."
            )
        scoring_metrics[ours] = float(value)
    return ScoringResult(summary=_summarize(battles), metrics=scoring_metrics)
