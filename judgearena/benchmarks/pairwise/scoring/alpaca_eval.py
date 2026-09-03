"""Local AlpacaEval 2.0 raw and length-controlled scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn import config_context
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold

from judgearena.benchmarks.pairwise.scoring.metrics import PairwiseWinRateMetric
from judgearena.log import get_logger

logger = get_logger(__name__)


@cache
def _load_gamed_data(
    repo_id: str,
    filename: str,
    revision: str,
) -> pd.DataFrame:
    """Load the task-configured AlpacaEval length-control calibration rows."""
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
    )
    return pd.read_csv(path).drop(columns="model")


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
    return annotations.reset_index(drop=True)


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


def _length_controlled_metrics(
    annotations: pd.DataFrame,
    *,
    calibration_repo_id: str,
    calibration_filename: str,
    calibration_revision: str,
    gamed_weight: float,
) -> dict[str, float]:
    """Reproduce AlpacaEval 2.0's length-controlled GLM metrics."""
    models = annotations["generator_2"].unique().tolist()
    baselines = annotations["generator_1"].unique().tolist()
    if len(models) != 1 or len(baselines) != 1:
        raise ValueError("AlpacaEval scoring requires one candidate and one baseline.")

    targets = annotations["preference"].astype(float) - 1
    parsed_targets = targets.dropna()
    if (~np.isfinite(parsed_targets) | ~parsed_targets.between(0, 1)).any():
        raise ValueError(
            "Normalized AlpacaEval preferences must be finite values in [0, 1]."
        )
    raw_winrate = float(parsed_targets.mean() * 100)
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

    gamed = _load_gamed_data(
        calibration_repo_id,
        calibration_filename,
        calibration_revision,
    )
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
    # Upstream fits only parsed labels but predicts every live instruction.
    # This keeps the LC population fixed when a judge response is unparseable.
    training = pd.concat([gamed, live.dropna(subset=["preference"])], ignore_index=True)
    sample_weight = np.where(
        training["not_gamed_baseline"],
        1.0,
        gamed_weight,
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


@dataclass(frozen=True, kw_only=True)
class AlpacaEvalLengthControlledMetric:
    """Official AlpacaEval 2.0 local length-controlled score."""

    calibration_repo_id: str
    calibration_filename: str
    calibration_revision: str
    gamed_weight: float

    def __post_init__(self) -> None:
        for name in (
            "calibration_repo_id",
            "calibration_filename",
            "calibration_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if (
            not isinstance(self.gamed_weight, int | float)
            or not math.isfinite(self.gamed_weight)
            or self.gamed_weight < 0
        ):
            raise ValueError("gamed_weight must be a finite non-negative number.")

    def calculate(self, battles: pd.DataFrame) -> dict[str, object]:
        if battles["instruction_index"].duplicated().any():
            logger.warning(
                "Battles contain duplicate instructions (swap_mode='both'?); the "
                "official LC protocol judges each instruction once in randomized order."
            )

        upstream = _length_controlled_metrics(
            _official_annotations(battles),
            calibration_repo_id=self.calibration_repo_id,
            calibration_filename=self.calibration_filename,
            calibration_revision=self.calibration_revision,
            gamed_weight=self.gamed_weight,
        )
        result: dict[str, object] = {
            **PairwiseWinRateMetric().calculate(battles),
        }
        for name, upstream_name in (
            ("length_controlled_winrate", "length_controlled_winrate"),
            ("lc_standard_error", "lc_standard_error"),
            ("raw_winrate", "win_rate"),
        ):
            value = upstream.get(upstream_name)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(
                    "Official AlpacaEval scoring did not return a finite "
                    f"{upstream_name!r}."
                )
            result[name] = float(value) / 100
        return result

    @staticmethod
    def render(result: dict[str, object]) -> str:
        return (
            "alpaca_eval_length_controlled: "
            f"{result['length_controlled_winrate']:.2%} "
            f"(raw {result['raw_winrate']:.2%}, "
            f"SE {result['lc_standard_error']:.2%})"
        )
