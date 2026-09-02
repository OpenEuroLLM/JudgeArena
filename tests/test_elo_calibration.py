"""Tests for PairScore temperature calibration."""

from copy import deepcopy

import numpy as np
import pandas as pd

from judgearena.benchmarks.elo.calibration import (
    calibrate_pairscore_temperature,
    fit_temperature,
)


def test_fit_temperature_follows_human_preference_direction():
    score_differences = np.array([2.0, 1.0, -1.0, -2.0])
    outcomes = np.array([1.0, 1.0, 0.0, 0.0])

    assert fit_temperature(score_differences, outcomes) > 0
    assert fit_temperature(score_differences, 1 - outcomes) < 0


def test_disabled_calibration_does_not_consume_rng():
    rng = np.random.default_rng(7)
    state = deepcopy(rng.bit_generator.state)

    result = calibrate_pairscore_temperature(
        pd.DataFrame(),
        pd.DataFrame(),
        enabled=False,
        soft_elo=True,
        sample_size=None,
        rng=rng,
        judge_model="unused",
        judge_model_kwargs={},
        swap_mode="fixed",
        prompt=None,  # type: ignore[arg-type]
        truncate_input_chars=None,
        default_temperature=0.3,
    )

    assert result is None
    assert rng.bit_generator.state == state
