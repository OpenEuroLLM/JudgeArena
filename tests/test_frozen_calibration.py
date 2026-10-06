"""Frozen PairScore calibration keeps physical pairs and both judge orders."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from judgearena.benchmarks.elo import calibration
from judgearena.config import RunConfig
from judgearena.prompts.parsing import PairScore


def _config():
    return RunConfig(
        task="elo-comparia",
        model={"name": "Dummy/candidate"},
        judge={"model": "Dummy/score A: 6 score B: 4", "swap_mode": "fixed"},
        elo={"soft_elo": True, "calibrate_temperature": True},
        run={"seed": 7, "store_root": None},
    )


def _battles():
    count = 12
    return pd.DataFrame(
        {
            "question_id": [f"q{i}" for i in range(count)],
            "lang": ["en"] * count,
            "model_a": ["reference"] * count,
            "model_b": ["strong"] * count,
            "winner": ["model_a" if i % 4 else "model_b" for i in range(count)],
            "conversation_a": [
                [{"content": f"prompt {i}"}, {"content": f"a {i}"}]
                for i in range(count)
            ],
            "conversation_b": [
                [{"content": f"prompt {i}"}, {"content": f"b {i}"}]
                for i in range(count)
            ],
        }
    )


def _prompt():
    return SimpleNamespace(
        parser=PairScore(),
        system_prompt="Judge both answers.",
        user_prompt_template="{user_prompt} A: {completion_A} B: {completion_B}",
        preset_name=None,
    )


def _calibrate(cfg, battles):
    calibration.calibrate_frozen_temperature(
        cfg,
        battles,
        ["reference", "strong"],
        ["en"],
        _prompt(),
        arena="comparia",
    )


def _annotation(a=None, b=None):
    return SimpleNamespace(
        parsed=None if a is None else SimpleNamespace(scores={"A": a, "B": b})
    )


def test_frozen_fit_averages_pass_probabilities_but_ordinary_fit_uses_direct(
    monkeypatch,
):
    cfg = _config()
    cfg.judge.swap_mode = "both"
    battles = _battles()
    monkeypatch.setattr(
        calibration,
        "judge_and_parse_prefs",
        lambda **_kwargs: (
            [_annotation(6, 4)] * len(battles),
            [_annotation(3, 7)] * len(battles),
            pd.Series(dtype=float),
        ),
    )

    _calibrate(cfg, battles)
    beta = cfg.elo.soft_elo_temperature
    # After reversing the swapped pass, both orders favor A, with gaps 2 and 4.
    assert np.mean(1 / (1 + np.exp(-beta * np.array([2, 4])))) == pytest.approx(0.75)
    assert beta != pytest.approx(np.log(3) / 3)  # Not sigmoid of the mean gap.

    ordinary = calibration.calibrate_pairscore_temperature(
        battles,
        battles,
        enabled=True,
        soft_elo=True,
        sample_size=None,
        rng=np.random.default_rng(cfg.run.seed),
        judge_model=cfg.judge.model,
        judge_model_kwargs={},
        swap_mode=cfg.judge.swap_mode,
        prompt=_prompt(),
        truncate_input_chars=None,
        default_temperature=0.3,
        arena="comparia",
    )
    assert ordinary == pytest.approx(np.log(3) / 2)


def test_calibration_retains_one_usable_pass_but_drops_missing_pairs_and_ties():
    differences, outcomes = calibration._calibration_data(
        [_annotation(6, 4), _annotation(), _annotation(), _annotation(6, 4)],
        [_annotation(), _annotation(4, 6), _annotation(), _annotation(4, 6)],
        ["model_a", "model_b", "model_a", "tie"],
    )
    np.testing.assert_allclose(differences, [[-2, np.nan], [np.nan, -2]])
    assert outcomes == [0, 1]


def test_fitted_beta_reuses_native_inference_cache(tmp_path, monkeypatch):
    import judgearena.models as models

    cfg = _config()
    cfg.run.store_root = tmp_path
    second = cfg.model_copy(deep=True)
    _calibrate(cfg, _battles())
    assert cfg.elo.soft_elo_temperature == pytest.approx(np.log(3) / 2)
    assert cfg.elo.calibrate_temperature is False
    assert cfg.elo.calibration_size is None

    def fail_build(*_args, **_kwargs):
        pytest.fail("A full inference-cache hit must not create a backend")

    monkeypatch.setattr(models, "make_model", fail_build)
    _calibrate(second, _battles())
    assert second.elo.soft_elo_temperature == cfg.elo.soft_elo_temperature
