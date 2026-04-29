"""Verify per-role sampling parameters reach the underlying backend.

Uses the ``Dummy`` provider (which records every kwarg the constructor sees
on :attr:`DummyModel.init_kwargs`) so we can assert that ``--seed_A``,
``--temperature_A`` etc. are actually forwarded to the model layer rather
than dropped on the floor by the CLI dispatcher.
"""

from __future__ import annotations

from judgearena.cli_common import GenerationConfig, gen_config_to_invoke_kwargs
from judgearena.utils import make_model


def test_make_model_dummy_captures_temperature_and_seed():
    """Dummy backend records the constructor kwargs so tests can verify them."""
    model = make_model("Dummy/foo", max_tokens=64, temperature=0.3, seed=42)
    assert model.init_kwargs.get("temperature") == 0.3
    assert model.init_kwargs.get("seed") == 42


def test_make_model_dummy_forwards_top_p_and_top_k():
    model = make_model("Dummy/foo", max_tokens=64, top_p=0.95, top_k=50)
    assert model.init_kwargs.get("top_p") == 0.95
    assert model.init_kwargs.get("top_k") == 50


def test_make_model_dummy_forwards_engine_kwargs():
    """Arbitrary engine-specific kwargs flow through to DummyModel.init_kwargs."""
    model = make_model("Dummy/foo", max_tokens=64, my_extra_flag="hello")
    assert model.init_kwargs.get("my_extra_flag") == "hello"


def test_gen_config_to_invoke_kwargs_skips_none():
    """Unset fields (None) on GenerationConfig are not forwarded."""
    cfg = GenerationConfig(temperature=None, seed=None, top_p=None, top_k=None)
    kwargs = gen_config_to_invoke_kwargs(cfg)
    # Only the always-present max_tokens plus whatever was explicitly set.
    assert kwargs == {"max_tokens": cfg.max_tokens}


def test_gen_config_to_invoke_kwargs_includes_set_fields():
    cfg = GenerationConfig(
        temperature=0.0,
        top_p=0.95,
        top_k=50,
        seed=7,
        max_tokens=128,
        max_model_len=8192,
        chat_template="<ct>",
        engine_kwargs={"tensor_parallel_size": 2},
    )
    kwargs = gen_config_to_invoke_kwargs(cfg)
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 0.95
    assert kwargs["top_k"] == 50
    assert kwargs["seed"] == 7
    assert kwargs["max_tokens"] == 128
    assert kwargs["max_model_len"] == 8192
    assert kwargs["chat_template"] == "<ct>"
    assert kwargs["tensor_parallel_size"] == 2


def test_per_role_seed_only_reaches_correct_model(monkeypatch):
    """A per-role --seed_A flag must not bleed into model B / judge."""
    from judgearena import cli as cli_module

    captured_args: dict = {}

    def fake_main_ge(args) -> None:
        captured_args["args"] = args

    monkeypatch.setattr(cli_module, "main_generate_and_evaluate", fake_main_ge)

    cli_module.cli(
        [
            "--task",
            "alpaca-eval",
            "--model_A",
            "Dummy/A",
            "--model_B",
            "Dummy/B",
            "--judge",
            "Dummy/J",
            "--seed_A",
            "11",
            "--temperature_judge",
            "0.0",
        ]
    )
    args = captured_args["args"]
    assert args.gen_A.seed == 11
    assert args.gen_B.seed is None
    assert args.gen_judge.seed is None
    assert args.gen_judge.temperature == 0.0
    assert args.gen_A.temperature is None
