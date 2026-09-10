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


def test_fit_temperature_follows_human_preference_direction():
    score_differences = np.array([2.0, 1.0, -1.0, -2.0])
    outcomes = np.array([1.0, 1.0, 0.0, 0.0])

    assert fit_temperature(score_differences, outcomes) > 0
    assert fit_temperature(score_differences, 1 - outcomes) < 0


@pytest.mark.parametrize(
    ("enabled", "prompt"),
    [
        (False, None),
        (True, SimpleNamespace(parser=object())),
    ],
)
def test_skipped_calibration_does_not_consume_rng_or_build_judge(
    monkeypatch, enabled, prompt
):
    rng = np.random.default_rng(7)
    state = deepcopy(rng.bit_generator.state)

    def fail_if_called(**_kwargs):
        raise AssertionError("calibration judge was built")

    monkeypatch.setattr(elo_calibration, "make_model", fail_if_called)
    result = calibrate_pairscore_temperature(
        pd.DataFrame(),
        pd.DataFrame(),
        enabled=enabled,
        soft_elo=True,
        sample_size=None,
        rng=rng,
        judge_model="unused",
        judge_model_kwargs={},
        swap_mode="fixed",
        prompt=prompt,
        truncate_input_chars=None,
        default_temperature=0.3,
    )

    assert result is None
    assert rng.bit_generator.state == state
