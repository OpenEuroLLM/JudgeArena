"""Tests for PairScore temperature calibration."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from judgearena.benchmarks.elo import calibration as elo_calibration
from judgearena.benchmarks.elo.calibration import (
    calibrate_pairscore_temperature,
    fit_temperature,
)


def test_fit_temperature_matches_observed_odds():
    # P(A>B) = 3/4 and score_A - score_B = -1 imply T = -log(3).
    score_differences = np.full(4, -1.0)
    outcomes = np.array([1.0, 1.0, 1.0, 0.0])

    assert fit_temperature(score_differences, outcomes) == pytest.approx(-np.log(3))


def test_non_pairscore_calibration_does_not_consume_rng_or_build_judge(monkeypatch):
    rng = np.random.default_rng(7)
    state = deepcopy(rng.bit_generator.state)

    def fail_if_called(**_kwargs):
        raise AssertionError("calibration judge was built")

    monkeypatch.setattr(elo_calibration, "prepare_model", fail_if_called)
    result = calibrate_pairscore_temperature(
        pd.DataFrame(),
        pd.DataFrame(),
        enabled=True,
        soft_elo=True,
        sample_size=None,
        rng=rng,
        judge_model="unused",
        judge_model_kwargs={},
        swap_mode="fixed",
        prompt=SimpleNamespace(parser=object()),
        truncate_input_chars=None,
        default_temperature=0.3,
        arena="test-arena",
    )

    assert result is None
    assert rng.bit_generator.state == state
