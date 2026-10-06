"""PairScore temperature calibration against human arena preferences."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar
from scipy.special import expit

from judgearena.arenas_utils import extract_turn_text
from judgearena.benchmarks.elo.rating import winner_to_pref
from judgearena.benchmarks.execution import build_judge
from judgearena.cache.inference import JudgementInferenceCache
from judgearena.evaluate import judge_and_parse_prefs
from judgearena.log import get_logger
from judgearena.models import prepare_model
from judgearena.prompts.parsing import PairScore
from judgearena.prompts.registry import ResolvedJudgePrompt

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


def fit_temperature(
    delta_s: ArrayLike,
    y: ArrayLike,
    bounds: tuple[float, float] = (-10.0, 10.0),
) -> float:
    """Fit a signed beta from score gaps and matching binary outcomes.

    A 1D input fits each direct pass with stable logistic NLL. An N-by-pass
    input fits each physical pair's mean probability across finite passes,
    not the probability of its mean score gap. Each row must have a usable
    pass; this mode also rejects scores that cannot identify beta.
    Ties (y == 0.5) are excluded. The caller decides whether beta must be positive.
    """
    delta_s = np.asarray(delta_s, dtype=float)
    y = np.asarray(y, dtype=float)
    if delta_s.ndim not in (1, 2) or len(delta_s) != len(y):
        raise ValueError("Score differences and outcomes must contain the same rows.")
    non_tie = y != 0.5
    delta_s = delta_s[non_tie]
    y = y[non_tie]
    if len(delta_s) == 0:
        raise ValueError(
            "No non-tie observations available for temperature calibration."
        )

    if delta_s.ndim == 1:
        agreement = (2 * y - 1) * delta_s

        def negative_log_likelihood(beta: float) -> float:
            return float(np.sum(np.logaddexp(0.0, -beta * agreement)))

    else:
        delta_s = np.where(np.isfinite(delta_s), delta_s, np.nan)

        def probabilities(beta: float) -> np.ndarray:
            return np.nanmean(expit(beta * delta_s), axis=1)

        def negative_log_likelihood(beta: float) -> float:
            predicted = np.clip(probabilities(beta), 1e-12, 1 - 1e-12)
            return float(
                -np.sum(y * np.log(predicted) + (1 - y) * np.log1p(-predicted))
            )

        lower, upper = bounds
        probe = np.stack(
            [
                probabilities(lower),
                probabilities((lower + upper) / 2),
                probabilities(upper),
            ]
        )
        if np.all(np.ptp(probe, axis=0) <= 1e-10):
            raise ValueError("Judge scores do not identify a soft-Elo beta.")

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=bounds,
        method="bounded",
    )
    return float(result.x)


def _sample_calibration_battles(
    battles: pd.DataFrame,
    sample_size: int | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_samples = (
        min(sample_size, len(battles)) if sample_size is not None else len(battles)
    )
    return battles.sample(n=n_samples, random_state=int(rng.integers(0, 2**31)))


def _judge_calibration_battles(
    battles: pd.DataFrame,
    judge,
    *,
    swap_mode: str,
    prompt: ResolvedJudgePrompt,
    truncate_input_chars: int | None,
    arena: str,
    strip_thinking_before_judging: bool = False,
):
    """Judge sampled source rows with the same text and cache metadata in both paths."""
    return judge_and_parse_prefs(
        judge_chat_model=judge,
        instructions=[
            extract_turn_text(turns[0]) for turns in battles["conversation_a"]
        ],
        completions_A=[
            extract_turn_text(turns[1]) for turns in battles["conversation_a"]
        ],
        completions_B=[
            extract_turn_text(turns[1]) for turns in battles["conversation_b"]
        ],
        swap_mode=swap_mode,
        strip_thinking_before_judging=strip_thinking_before_judging,
        system_prompt=prompt.system_prompt,
        user_prompt_template=prompt.user_prompt_template,
        prompt_preset=prompt.preset_name,
        parse=prompt.parser,
        truncate_input_chars=truncate_input_chars,
        cache_row_metadata=[
            {
                "instruction_id": f"{arena}:{row.question_id}",
                "model_a": row.model_a,
                "model_b": row.model_b,
                "orientation": "direct",
            }
            for row in battles.itertuples()
        ],
    )


def _score_difference(annotation) -> float | None:
    """Return the presented A-minus-B gap, or None when a score is missing."""
    scores = {} if annotation.parsed is None else annotation.parsed.scores
    score_a, score_b = scores.get("A"), scores.get("B")
    return None if score_a is None or score_b is None else score_a - score_b


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
    arena: str,
    inference_cache: JudgementInferenceCache | None = None,
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
    calibration_battles = _sample_calibration_battles(arena_battles, sample_size, rng)
    calibration_judge = prepare_model(
        model=judge_model,
        cache=inference_cache,
        **dict(judge_model_kwargs),
    )
    annotations, _, _ = _judge_calibration_battles(
        source_battles.loc[calibration_battles.index],
        calibration_judge,
        swap_mode=swap_mode,
        prompt=prompt,
        truncate_input_chars=truncate_input_chars,
        arena=arena,
    )

    score_differences: list[float] = []
    outcomes: list[float] = []
    for annotation, human_winner in zip(
        annotations, calibration_battles["winner"].tolist(), strict=True
    ):
        difference = _score_difference(annotation)
        if difference is None:
            continue
        human_preference = winner_to_pref(human_winner)
        if human_preference is None or human_preference == 0.5:
            continue
        score_differences.append(difference)
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


def _calibration_data(
    annotations,
    reversed_annotations,
    human_winners: list[str],
) -> tuple[list[list[float]], list[float]]:
    """Return canonical B-minus-A score gaps and human B outcomes."""
    score_differences: list[list[float]] = []
    outcomes: list[float] = []
    for index, human_winner in enumerate(human_winners):
        human_preference = winner_to_pref(human_winner)
        if human_preference is None or human_preference == 0.5:
            continue

        direct_difference = _score_difference(annotations[index])
        differences = [np.nan if direct_difference is None else -direct_difference]
        if reversed_annotations is not None:
            reversed_difference = _score_difference(reversed_annotations[index])
            differences.append(
                np.nan if reversed_difference is None else reversed_difference
            )
        if np.isfinite(differences).any():
            score_differences.append(differences)
            outcomes.append(human_preference)
    return score_differences, outcomes


def calibrate_frozen_temperature(
    cfg: RunConfig,
    battles: pd.DataFrame,
    anchors: list[str],
    languages: list[str],
    resolved_prompt: ResolvedJudgePrompt,
    *,
    arena: str,
) -> None:
    """Resolve one soft-Elo beta before the leaderboard is frozen."""
    assert cfg.elo is not None
    if not cfg.elo.soft_elo:
        if cfg.elo.calibrate_temperature:
            raise ValueError("soft-Elo calibration requires elo.soft_elo: true.")
        return
    if not cfg.elo.calibrate_temperature:
        beta = cfg.elo.soft_elo_temperature
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("elo.soft_elo_temperature must be finite and positive.")
        return

    calibration_battles = battles.loc[
        battles["lang"].isin(languages)
        & battles["model_a"].isin(anchors)
        & battles["model_b"].isin(anchors)
        & battles["model_a"].ne(battles["model_b"])
    ]
    if not isinstance(resolved_prompt.parser, PairScore):
        raise ValueError("Frozen soft-Elo calibration requires a PairScore parser.")
    calibration_battles = _sample_calibration_battles(
        calibration_battles,
        cfg.elo.calibration_size,
        np.random.default_rng(cfg.run.seed),
    )
    annotations, reversed_annotations, _ = _judge_calibration_battles(
        calibration_battles,
        build_judge(cfg),
        swap_mode=cfg.judge.swap_mode,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        prompt=resolved_prompt,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        arena=arena,
    )
    score_differences, outcomes = _calibration_data(
        annotations, reversed_annotations, calibration_battles["winner"].tolist()
    )
    if len(score_differences) < 10:
        raise ValueError(
            "Frozen soft-Elo calibration needs at least 10 usable physical pairs; "
            f"found {len(score_differences)}."
        )
    beta = fit_temperature(score_differences, outcomes)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError(
            "The frozen judge did not produce a finite positive soft-Elo beta."
        )
    cfg.elo = cfg.elo.model_copy(
        update={
            "soft_elo_temperature": beta,
            "calibrate_temperature": False,
            "calibration_size": None,
        }
    )
