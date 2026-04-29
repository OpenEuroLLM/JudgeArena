"""Tests for the unified `judgearena` CLI dispatcher."""

from __future__ import annotations

import pytest

from judgearena import cli as cli_module
from judgearena.estimate_elo_ratings import CliEloArgs
from judgearena.generate_and_evaluate import CliArgs


@pytest.fixture
def capture_mains(monkeypatch):
    """Replace both main functions with spies that record the dispatched args."""
    captured: dict[str, object] = {}

    def fake_main_ge(args: CliArgs) -> None:
        captured["module"] = "generate_and_evaluate"
        captured["args"] = args

    def fake_main_elo(args: CliEloArgs) -> None:
        captured["module"] = "elo"
        captured["args"] = args

    monkeypatch.setattr(cli_module, "main_generate_and_evaluate", fake_main_ge)
    monkeypatch.setattr(cli_module, "main_elo", fake_main_elo)
    return captured


@pytest.mark.parametrize(
    "task",
    [
        "alpaca-eval",
        "arena-hard-v2.0",
        "m-arena-hard-EU",
        "fluency-french",
        "mt-bench",
    ],
)
def test_task_dispatches_to_generate_and_evaluate(capture_mains, task: str):
    cli_module.cli(
        [
            "--task",
            task,
            "--model_A",
            "Dummy/A",
            "--model_B",
            "Dummy/B",
            "--judge",
            "Dummy/J",
        ]
    )
    assert capture_mains["module"] == "generate_and_evaluate"
    ge_args: CliArgs = capture_mains["args"]
    assert isinstance(ge_args, CliArgs)
    assert ge_args.task == task
    assert ge_args.model_A == "Dummy/A"
    assert ge_args.model_B == "Dummy/B"
    assert ge_args.judge_model == "Dummy/J"


@pytest.mark.parametrize(
    "task, expected_arena",
    [
        ("elo-comparia", "ComparIA"),
        ("elo-lmarena-140k", "LMArena-140k"),
        ("elo-lmarena-100k", "LMArena-100k"),
        ("elo-lmarena", "LMArena"),
    ],
)
def test_elo_task_dispatches_to_estimate_elo_ratings(
    capture_mains, task: str, expected_arena: str
):
    cli_module.cli(
        [
            "--task",
            task,
            "--model_A",
            "Dummy/X",
            "--judge",
            "Dummy/J",
        ]
    )
    assert capture_mains["module"] == "elo"
    elo_args: CliEloArgs = capture_mains["args"]
    assert isinstance(elo_args, CliEloArgs)
    assert elo_args.arena == expected_arena
    assert elo_args.model == "Dummy/X"
    assert elo_args.judge_model == "Dummy/J"


def test_dataset_flag_is_deprecated_alias(capture_mains):
    with pytest.warns(DeprecationWarning, match="--dataset is deprecated"):
        cli_module.cli(
            [
                "--dataset",
                "alpaca-eval",
                "--model_A",
                "Dummy/A",
                "--model_B",
                "Dummy/B",
                "--judge",
                "Dummy/J",
            ]
        )
    assert capture_mains["module"] == "generate_and_evaluate"
    assert capture_mains["args"].task == "alpaca-eval"


def test_arena_flag_is_deprecated_alias_lowercases(capture_mains):
    """`--arena ComparIA` must still route to the ComparIA ELO path.

    The deprecated path is allowed to be case-insensitive because that was the
    historical contract for ``judgearena-elo --arena LMArena-140k``.
    """
    with pytest.warns(DeprecationWarning, match="--arena is deprecated"):
        cli_module.cli(
            [
                "--arena",
                "ComparIA",
                "--model_A",
                "Dummy/X",
                "--judge",
                "Dummy/J",
            ]
        )
    assert capture_mains["module"] == "elo"
    assert capture_mains["args"].arena == "ComparIA"


def test_model_flag_is_deprecated_alias(capture_mains):
    with pytest.warns(DeprecationWarning, match="--model is deprecated"):
        cli_module.cli(
            [
                "--task",
                "elo-comparia",
                "--model",
                "Dummy/X",
                "--judge",
                "Dummy/J",
            ]
        )
    assert capture_mains["module"] == "elo"
    assert capture_mains["args"].model == "Dummy/X"


def test_model_and_model_a_collide_raises(capture_mains):
    with pytest.raises(SystemExit, match="exactly one of --model_A/--model"):
        cli_module.cli(
            [
                "--task",
                "elo-comparia",
                "--model",
                "Dummy/X",
                "--model_A",
                "Dummy/Y",
                "--judge",
                "Dummy/J",
            ]
        )


def test_missing_task_raises(capture_mains):
    with pytest.raises(SystemExit, match="One of --task"):
        cli_module.cli(
            [
                "--model_A",
                "Dummy/A",
                "--model_B",
                "Dummy/B",
                "--judge",
                "Dummy/J",
            ]
        )


def test_multiple_task_flags_raise(capture_mains):
    with pytest.raises(SystemExit, match="exactly one of"):
        cli_module.cli(
            [
                "--task",
                "alpaca-eval",
                "--dataset",
                "alpaca-eval",
                "--model_A",
                "Dummy/A",
                "--model_B",
                "Dummy/B",
                "--judge",
                "Dummy/J",
            ]
        )


def test_elo_task_requires_model_a(capture_mains):
    with pytest.raises(SystemExit, match="--model_A is required for elo"):
        cli_module.cli(
            [
                "--task",
                "elo-comparia",
                "--judge",
                "Dummy/J",
            ]
        )


def test_elo_task_rejects_model_b_for_now(capture_mains):
    with pytest.raises(SystemExit, match="not yet supported"):
        cli_module.cli(
            [
                "--task",
                "elo-comparia",
                "--model_A",
                "Dummy/A",
                "--model_B",
                "Dummy/B",
                "--judge",
                "Dummy/J",
            ]
        )


def test_uppercase_elo_task_is_rejected(capture_mains):
    with pytest.raises(SystemExit, match="Unknown elo task"):
        cli_module.cli(
            [
                "--task",
                "elo-LMArena-140k",
                "--model_A",
                "Dummy/X",
                "--judge",
                "Dummy/J",
            ]
        )


def test_unknown_elo_task_raises(capture_mains):
    with pytest.raises(SystemExit, match="Unknown elo task.*elo-lmarena-140k"):
        cli_module.cli(
            [
                "--task",
                "elo-foo",
                "--model_A",
                "Dummy/X",
                "--judge",
                "Dummy/J",
            ]
        )


def test_generate_and_evaluate_requires_model_a_and_b(capture_mains):
    with pytest.raises(SystemExit, match="--model_A and --model_B are required"):
        cli_module.cli(
            [
                "--task",
                "alpaca-eval",
                "--model_A",
                "Dummy/A",
                "--judge",
                "Dummy/J",
            ]
        )


def test_deprecated_model_flag_routes_into_pairwise_task(capture_mains):
    """`--model` is a deprecated alias for `--model_A` even on pairwise tasks."""
    with pytest.warns(DeprecationWarning, match="--model is deprecated"):
        cli_module.cli(
            [
                "--task",
                "alpaca-eval",
                "--model",
                "Dummy/A",
                "--model_B",
                "Dummy/B",
                "--judge",
                "Dummy/J",
            ]
        )
    assert capture_mains["module"] == "generate_and_evaluate"
    assert capture_mains["args"].model_A == "Dummy/A"
    assert capture_mains["args"].model_B == "Dummy/B"


def test_elo_forwards_optional_flags(capture_mains):
    cli_module.cli(
        [
            "--task",
            "elo-lmarena-140k",
            "--model_A",
            "Dummy/X",
            "--judge",
            "Dummy/J",
            "--languages",
            "en",
            "fr",
            "--n_instructions_per_language",
            "50",
            "--n_bootstraps",
            "5",
            "--seed",
            "7",
            "--baseline_model",
            "gpt-4o",
        ]
    )
    elo_args: CliEloArgs = capture_mains["args"]
    assert elo_args.languages == ["en", "fr"]
    assert elo_args.n_instructions_per_language == 50
    assert elo_args.n_bootstraps == 5
    assert elo_args.seed == 7
    assert elo_args.baseline_model == "gpt-4o"


def test_engine_kwargs_parsed_as_json(capture_mains):
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
            "--engine_kwargs",
            '{"tensor_parallel_size": 4}',
        ]
    )
    ge_args: CliArgs = capture_mains["args"]
    assert ge_args.engine_kwargs == {"tensor_parallel_size": 4}


def test_judge_prompt_preset_flag_is_forwarded(capture_mains):
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
            "--judge_prompt_preset",
            "default_with_explanation",
        ]
    )
    ge_args: CliArgs = capture_mains["args"]
    assert ge_args.judge_prompt_preset == "default_with_explanation"
    assert ge_args.judge_system_prompt_file is None
    assert ge_args.judge_user_prompt_file is None
