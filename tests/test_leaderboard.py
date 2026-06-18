"""Tests for the judgearena.leaderboard package (offline; judge calls mocked)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from judgearena.leaderboard.kappa import language_kappa


def test_language_kappa_perfect_agreement():
    judge = pd.Series([0.0, 1.0, 0.0, 1.0])
    human = pd.Series([0.0, 1.0, 0.0, 1.0])
    assert language_kappa(judge, human) == pytest.approx(1.0)


def test_language_kappa_chance_agreement_is_zero():
    judge = pd.Series([0.0, 0.0, 1.0, 1.0])
    human = pd.Series([0.0, 1.0, 0.0, 1.0])
    assert language_kappa(judge, human) == pytest.approx(0.0)


def test_language_kappa_excludes_ties_and_nan():
    # last two rows (a tie and a NaN) must be dropped, leaving perfect agreement
    judge = pd.Series([0.0, 1.0, 0.0, 1.0, 0.5, 1.0])
    human = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0, float("nan")])
    assert language_kappa(judge, human) == pytest.approx(1.0)


def test_language_kappa_single_class_returns_nan():
    judge = pd.Series([0.0, 0.0, 0.0])
    human = pd.Series([0.0, 0.0, 0.0])
    assert math.isnan(language_kappa(judge, human))
