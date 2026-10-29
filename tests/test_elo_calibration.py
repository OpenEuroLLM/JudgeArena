"""Tests for PairScore temperature calibration."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import judgearena.benchmarks.elo.calibration as calibration
from judgearena.benchmarks.elo.calibration import (
    calibrate_pairscore_temperature,
    fit_temperature,
)
from judgearena.prompts.registry import resolve_judge_prompt


def test_fit_temperature_matches_observed_odds():
    # P(A>B) = 3/4 and score_A - score_B = -1 imply T = -log(3).
    score_differences = np.full(4, -1.0)
    outcomes = np.array([1.0, 1.0, 1.0, 0.0])

    assert fit_temperature(score_differences, outcomes) == pytest.approx(-np.log(3))


def test_temperature_preserves_raw_annotations_and_uses_actual_swap():
    annotations = pd.DataFrame(
        {
            "parsed": [{"scores": {"A": 6, "B": 8}}] * 2 + [None],
            "pref": [0.75, 0.25, float("nan")],
            "swapped": [False, True, False],
            "orientation": "single",
        },
        index=[9, 2, 7],
    )
    original = annotations.copy(deep=True)

    outcomes = calibration.apply_pairscore_temperature(annotations, temperature=0.8)

    preference = 1 / (1 + np.exp(-1.6))
    assert outcomes.index.tolist() == [9, 2, 7]
    assert outcomes["pref"].tolist() == pytest.approx(
        [preference, 1 - preference, float("nan")], nan_ok=True
    )
    pd.testing.assert_frame_equal(annotations, original)


def test_non_pairscore_calibration_does_not_consume_rng_or_use_judge():
    rng = np.random.default_rng(7)
    state = deepcopy(rng.bit_generator.state)

    result = calibrate_pairscore_temperature(
        pd.DataFrame(),
        sample_size=None,
        rng=rng,
        judge_chat_model=object(),
        swap_mode="fixed",
        prompt=SimpleNamespace(parser=object()),
        truncate_input_chars=None,
        default_temperature=0.3,
    )

    assert result is None
    assert rng.bit_generator.state == state


@pytest.mark.parametrize("sample_size", [None, 100, 9])
def test_calibration_keeps_sampled_answers_and_labels_aligned(monkeypatch, sample_size):
    battles = pd.DataFrame(
        {
            "completion_a": [9, 8] * 6,
            "completion_b": [8, 9] * 6,
            "reference_pref": [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        },
        index=[10, 30, 50, 80, 90, 120, 140, 170, 200, 250, 300, 400],
    )

    def judge(_model, sample, _groups, **_kwargs):
        return pd.DataFrame(
            {
                "parsed": [
                    {"scores": {"A": row.completion_a, "B": row.completion_b}}
                    for row in sample.itertuples()
                ],
                "orientation": "single",
            }
        )

    monkeypatch.setattr(calibration, "judge_comparisons", judge)
    result = calibrate_pairscore_temperature(
        battles,
        sample_size=sample_size,
        rng=np.random.default_rng(7),
        judge_chat_model=object(),
        swap_mode="fixed",
        prompt=resolve_judge_prompt(preset="default"),
        truncate_input_chars=None,
        default_temperature=0.3,
    )

    # Nine agreements and three disagreements, each with a unit score gap.
    if sample_size == 9:
        assert result is None
    else:
        assert result == pytest.approx(np.log(3))
