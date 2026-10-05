"""Frozen PairScore calibration keeps physical pairs and both judge orders."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from judgearena.benchmarks.elo import freeze
from judgearena.config import RunConfig
from judgearena.prompts.parsing import PairScore


def _config(**elo):
    return RunConfig(
        task="elo-comparia",
        model={"name": "Dummy/candidate"},
        judge={"model": "Dummy/score A: 6 score B: 4", "swap_mode": "fixed"},
        elo={"soft_elo": True, "calibrate_temperature": True, **elo},
        run={"seed": 7, "store_root": None},
    )


def _battles(count=12):
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


def _calibrate(cfg, battles, prompt=None):
    freeze._calibrate_soft_elo(
        cfg,
        battles,
        ["reference", "strong"],
        ["en"],
        prompt or _prompt(),
        arena="comparia",
    )


def _annotation(a=None, b=None):
    return SimpleNamespace(
        parsed=None if a is None else SimpleNamespace(scores={"A": a, "B": b})
    )


def test_fit_temperature_matches_observed_odds():
    assert freeze._fit_temperature(
        np.full(4, -1.0), np.array([1.0, 1.0, 1.0, 0.0])
    ) == pytest.approx(-np.log(3))


def test_fit_temperature_averages_swap_probabilities():
    beta = 0.7
    differences = np.tile([1.0, 3.0], (20, 1))
    probability = np.mean(1.0 / (1.0 + np.exp(-beta * differences[0])))
    assert freeze._fit_temperature(
        differences, np.full(len(differences), probability)
    ) == pytest.approx(beta)


@pytest.mark.parametrize(
    "differences", [np.zeros((10, 1)), np.tile([1.0, -1.0], (10, 1))]
)
def test_fit_temperature_rejects_unidentifiable_scores(differences):
    with pytest.raises(ValueError, match="do not identify"):
        freeze._fit_temperature(differences, np.tile([0.0, 1.0], 5))


def test_calibration_uses_reoriented_scores_from_both_judge_passes(monkeypatch):
    cfg = _config()
    cfg.judge.swap_mode = "both"
    direct = [_annotation(6, 4)] * 12
    reversed_rows = [_annotation(3, 7)] * 12
    monkeypatch.setattr(freeze, "build_judge", lambda _cfg: object())
    monkeypatch.setattr(
        freeze,
        "judge_and_parse_prefs",
        lambda **_kwargs: (direct, reversed_rows, pd.Series(dtype=float)),
    )
    fitted = {}

    def fake_fit(differences, outcomes):
        fitted["differences"] = differences
        fitted["outcomes"] = outcomes
        return 0.75

    monkeypatch.setattr(freeze, "_fit_temperature", fake_fit)
    _calibrate(cfg, _battles().assign(winner="model_a"))
    assert cfg.elo.soft_elo_temperature == 0.75
    np.testing.assert_array_equal(fitted["differences"], [[-2, -4]] * 12)
    np.testing.assert_array_equal(fitted["outcomes"], [0] * 12)


def test_calibration_retains_one_usable_pass_but_drops_missing_pairs_and_ties():
    differences, outcomes = freeze._calibration_data(
        [
            _annotation(6, 4),
            _annotation(),
            _annotation(),
            _annotation(6, 4),
            _annotation(6, 4),
        ],
        [
            _annotation(),
            _annotation(3, 7),
            _annotation(),
            _annotation(3, 7),
            _annotation(3, 7),
        ],
        ["model_a", "model_b", "model_a", "tie", "unknown"],
    )
    np.testing.assert_allclose(differences, [[-2, np.nan], [np.nan, -4]])
    assert outcomes == [0, 1]
    assert freeze._fit_temperature(
        np.tile([[-1, np.nan], [np.nan, -1]], (2, 1)),
        [1, 1, 1, 0],
    ) == pytest.approx(-np.log(3))


@pytest.mark.parametrize("count", [9, 10])
def test_calibration_needs_ten_usable_physical_pairs(monkeypatch, count):
    cfg = _config()
    cfg.judge.swap_mode = "both"
    monkeypatch.setattr(freeze, "build_judge", lambda _cfg: object())
    monkeypatch.setattr(
        freeze,
        "judge_and_parse_prefs",
        lambda **_kwargs: (
            [_annotation()]
            + [_annotation(6, 4)] * (count - 1)
            + [_annotation()] * (12 - count),
            [_annotation(3, 7)] + [_annotation()] * 11,
            pd.Series(dtype=float),
        ),
    )
    if count < 10:
        with pytest.raises(
            ValueError, match=f"10 usable physical pairs; found {count}"
        ):
            _calibrate(cfg, _battles())
        assert cfg.elo.calibrate_temperature is True
    else:
        _calibrate(cfg, _battles())
        assert cfg.elo.soft_elo_temperature > 0
        assert cfg.elo.calibrate_temperature is False


@pytest.mark.parametrize("case", ["unidentifiable", "negative"])
def test_invalid_judge_fit_fails_closed(case):
    cfg = _config()
    battles = _battles()
    if case == "unidentifiable":
        cfg.judge.swap_mode = "both"
        message = "do not identify"
    else:
        battles["winner"] = battles["winner"].map(
            {"model_a": "model_b", "model_b": "model_a"}
        )
        message = "finite positive"
    with pytest.raises(ValueError, match=message):
        _calibrate(cfg, battles)
    assert cfg.elo.calibrate_temperature is True


@pytest.mark.parametrize("beta", [0, -0.3, np.nan, np.inf])
@pytest.mark.parametrize("calibrate", [False, True])
def test_calibration_rejects_nonpositive_or_nonfinite_beta(
    monkeypatch, beta, calibrate
):
    cfg = _config(calibrate_temperature=calibrate, soft_elo_temperature=beta)
    monkeypatch.setattr(freeze, "_fit_temperature", lambda *_args: beta)
    with pytest.raises(ValueError, match="finite.*positive"):
        _calibrate(cfg, _battles())


@pytest.mark.parametrize("case", ["hard", "parser"])
def test_unsupported_calibration_fails_before_building_judge(monkeypatch, case):
    cfg = _config(soft_elo=case != "hard")
    prompt = _prompt()
    if case == "parser":
        prompt.parser = object()

    def fail_build(_cfg):
        pytest.fail("Unsupported calibration must not build a judge")

    monkeypatch.setattr(freeze, "build_judge", fail_build)
    with pytest.raises(ValueError, match="requires"):
        _calibrate(cfg, _battles(), prompt)


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
