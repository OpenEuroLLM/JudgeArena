"""Verify per-role sampling parameters reach the underlying backend."""

from __future__ import annotations

from judgearena.config import RunConfig, build_run_config
from judgearena.models import make_model
from judgearena.utils import generation_cache_token


def test_make_model_dummy_captures_temperature_and_seed():
    """Dummy backend records the constructor kwargs so tests can verify them."""
    model = make_model("Dummy/foo", max_tokens=64, temperature=0.3, seed=42)
    assert model.init_kwargs.get("temperature") == 0.3
    assert model.init_kwargs.get("seed") == 42


def test_dedicated_sampling_fields_only_override_when_set():
    cfg = RunConfig(
        task="alpaca-eval-ja",
        model={"name": "Dummy/a", "engine_kwargs": {"temperature": 0.5}},
        judge={
            "model": "Dummy/j",
            "temperature": 0.3,
            "engine_kwargs": {"temperature": 0.2},
        },
    )
    assert cfg.model.evaluated_generation_kwargs()["temperature"] == 0.5
    assert cfg.judge.model_kwargs()["temperature"] == 0.3


def test_baseline_sampling_params_inherit_from_model_when_unset():
    cfg = RunConfig(
        task="alpaca-eval-ja",
        model={
            "name": "Dummy/a",
            "baseline": "Dummy/b",
            "temperature": 0.3,
            "seed": 7,
            "baseline_temperature": 0.95,
        },
        judge={"model": "Dummy/j"},
    )
    kwargs = cfg.model.baseline_generation_kwargs()
    assert kwargs["temperature"] == 0.95
    assert kwargs["seed"] == 7


def test_generation_cache_token_is_sensitive_to_sampling_params():
    base = {"max_tokens": 32768, "temperature": 0.0, "seed": 1}
    same = {"seed": 1, "temperature": 0.0, "max_tokens": 32768}  # order independent
    changed_seed = {**base, "seed": 2}

    assert generation_cache_token(base) == generation_cache_token(same)
    assert generation_cache_token(base) != generation_cache_token(changed_seed)


def test_model_args_per_role_kwargs_are_independent():
    cfg = RunConfig(
        task="alpaca-eval-ja",
        model={
            "name": "Dummy/a",
            "baseline": "Dummy/b",
            "seed": 11,
            "baseline_seed": 22,
            "baseline_max_out_tokens": 128,
            "engine_kwargs": {"shared": True},
            "baseline_engine_kwargs": {"baseline_only": True},
        },
        judge={"model": "Dummy/j", "seed": 33},
    )
    evaluated = cfg.model.evaluated_generation_kwargs()
    baseline = cfg.model.baseline_generation_kwargs()
    assert evaluated["seed"] == 11
    assert baseline["seed"] == 22
    assert cfg.judge.model_kwargs()["seed"] == 33
    assert evaluated["shared"] is True
    assert "shared" not in baseline
    assert baseline["baseline_only"] is True
    assert baseline["max_tokens"] == 128


def test_nested_cli_sampling_flags_land_on_correct_roles():
    cfg = build_run_config(
        [
            "--task",
            "alpaca-eval-ja",
            "--model.name",
            "Dummy/A",
            "--model.baseline",
            "Dummy/B",
            "--judge.model",
            "Dummy/J",
            "--model.seed",
            "11",
            "--model.baseline_seed",
            "22",
            "--judge.temperature",
            "0.0",
        ]
    )
    assert cfg.model.seed == 11
    assert cfg.model.baseline_seed == 22
    assert cfg.judge.temperature == 0.0
    assert cfg.model.temperature is None
