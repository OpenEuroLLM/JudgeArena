"""PairScore temperature calibration against human arena preferences."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from judgearena.benchmarks.pairwise.judging import judge_comparisons
from judgearena.log import get_logger
from judgearena.prompts.parsing import PairScore
from judgearena.prompts.registry import ResolvedJudgePrompt

logger = get_logger(__name__)


def fit_temperature(
    delta_s: np.ndarray,
    y: np.ndarray,
    bounds: tuple[float, float] = (-10.0, 10.0),
) -> float:
    """Fit ``T`` in ``P(A>B) = sigmoid(T * (score_A - score_B))``."""
    delta_s = np.asarray(delta_s, dtype=float)
    y = np.asarray(y, dtype=float)
    non_tie = y != 0.5
    delta_s = delta_s[non_tie]
    y = y[non_tie]
    if len(delta_s) == 0:
        raise ValueError(
            "No non-tie observations available for temperature calibration."
        )

    agreement = (2 * y - 1) * delta_s

    def negative_log_likelihood(temperature: float) -> float:
        return float(np.sum(np.logaddexp(0.0, -temperature * agreement)))

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=bounds,
        method="bounded",
    )
    return float(result.x)


def calibrate_pairscore_temperature(
    battles: pd.DataFrame,
    *,
    sample_size: int | None,
    rng: np.random.Generator,
    judge_chat_model,
    swap_mode: str,
    prompt: ResolvedJudgePrompt,
    truncate_input_chars: int | None,
    default_temperature: float,
) -> float | None:
    """Sample validated human comparisons and fit their PairScore temperature."""
    if not isinstance(prompt.parser, PairScore):
        parser_name = getattr(prompt.parser, "name", type(prompt.parser).__name__)
        logger.warning(
            "PairScore temperature calibration does not apply to parser %r; "
            "using its preferences unchanged.",
            parser_name,
        )
        return None

    logger.info("Calibrating PairScore temperature against human annotations.")
    n_samples = (
        min(sample_size, len(battles)) if sample_size is not None else len(battles)
    )
    sample = battles.sample(
        n=n_samples,
        random_state=int(rng.integers(0, 2**31)),
    )
    judgments = judge_comparisons(
        judge_chat_model,
        sample,
        [(prompt, sample.index)],
        swap_mode="both" if swap_mode == "both" else "fixed",
        strip_thinking_before_judging=False,
        truncate_input_chars=truncate_input_chars,
        use_tqdm=False,
    )
    # Calibration historically fits only the original arena answer order.
    annotations = judgments.loc[judgments["orientation"] != "reversed"]

    score_differences: list[float] = []
    outcomes: list[float] = []
    for parsed, human_preference in zip(
        annotations["parsed"], sample["pref"], strict=True
    ):
        scores = {} if parsed is None else parsed["scores"]
        score_a = scores.get("A")
        score_b = scores.get("B")
        if score_a is None or score_b is None:
            continue
        if human_preference not in (0.0, 1.0):
            continue
        score_differences.append(score_a - score_b)
        outcomes.append(1.0 - human_preference)

    if len(score_differences) < 10:
        logger.warning(
            "Only %d valid calibration pairs (need ≥10); keeping default temperature.",
            len(score_differences),
        )
        return None

    temperature = fit_temperature(
        np.array(score_differences),
        np.array(outcomes),
    )
    logger.info(
        "Calibration pairs: %d  T* = %.4f  (default was %s)",
        len(score_differences),
        temperature,
        default_temperature,
    )
    return temperature
