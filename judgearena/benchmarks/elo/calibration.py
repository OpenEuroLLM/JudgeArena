"""PairScore temperature calibration against human arena preferences."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from judgearena.arenas_utils import _extract_instruction_text
from judgearena.benchmarks.elo.rating import winner_to_pref
from judgearena.evaluate import judge_and_parse_prefs
from judgearena.log import get_logger
from judgearena.models import make_model
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
    arena_battles: pd.DataFrame,
    source_battles: pd.DataFrame,
    *,
    enabled: bool,
    soft_elo: bool,
    sample_size: int | None,
    rng: np.random.Generator,
    judge_model: str,
    judge_model_kwargs: Mapping[str, object],
    swap_mode: str,
    prompt: ResolvedJudgePrompt,
    truncate_input_chars: int | None,
    default_temperature: float,
) -> float | None:
    """Judge sampled human battles and return a fitted PairScore temperature."""
    if not enabled:
        return None
    if not soft_elo:
        logger.warning(
            "--calibrate-temperature has no effect with --no-soft-elo; skipping."
        )
        return None
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
        min(sample_size, len(arena_battles))
        if sample_size is not None
        else len(arena_battles)
    )
    calibration_battles = arena_battles.sample(
        n=n_samples,
        random_state=int(rng.integers(0, 2**31)),
    )
    instructions = [
        _extract_instruction_text(source_battles.loc[index, "conversation_a"][0])
        for index in calibration_battles.index
    ]
    completions_a = [
        _extract_instruction_text(source_battles.loc[index, "conversation_a"][1])
        for index in calibration_battles.index
    ]
    completions_b = [
        _extract_instruction_text(source_battles.loc[index, "conversation_b"][1])
        for index in calibration_battles.index
    ]

    calibration_judge = make_model(model=judge_model, **dict(judge_model_kwargs))
    annotations, _, _ = judge_and_parse_prefs(
        judge_chat_model=calibration_judge,
        instructions=instructions,
        completions_A=completions_a,
        completions_B=completions_b,
        swap_mode=swap_mode,
        system_prompt=prompt.system_prompt,
        user_prompt_template=prompt.user_prompt_template,
        prompt_preset=prompt.preset_name,
        parse=prompt.parser,
        truncate_input_chars=truncate_input_chars,
    )

    score_differences: list[float] = []
    outcomes: list[float] = []
    for annotation, human_winner in zip(
        annotations, calibration_battles["winner"].tolist(), strict=True
    ):
        scores = {} if annotation.parsed is None else annotation.parsed.scores
        score_a = scores.get("A")
        score_b = scores.get("B")
        if score_a is None or score_b is None:
            continue
        human_preference = winner_to_pref(human_winner)
        if human_preference is None or human_preference == 0.5:
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
