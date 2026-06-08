"""Tests for the unified `judgearena` CLI dispatcher."""

from __future__ import annotations

import pytest

from judgearena import cli as cli_module
from judgearena.config import RunConfig


@pytest.fixture
def capture_mains(monkeypatch):
    """Replace both main functions with spies that record the dispatched args."""
    captured: dict[str, object] = {}

    def fake_main_ge(args: RunConfig) -> None:
        captured["module"] = "generate_and_evaluate"
        captured["args"] = args

    def fake_main_elo(args: RunConfig) -> None:
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
        "m-arena-hard-v2.0-EU",
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
    ge_cfg: RunConfig = capture_mains["args"]
    assert isinstance(ge_cfg, RunConfig)
    assert ge_cfg.task == task
    assert ge_cfg.model.path == "Dummy/A"
    assert ge_cfg.model.path_b == "Dummy/B"
    assert ge_cfg.judge.model == "Dummy/J"


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
    elo_cfg: RunConfig = capture_mains["args"]
    assert isinstance(elo_cfg, RunConfig)
    assert elo_cfg.elo is not None
    assert elo_cfg.elo.arena == expected_arena
    assert elo_cfg.model.path == "Dummy/X"
    assert elo_cfg.judge.model == "Dummy/J"


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
    """`--arena ComparIA` must still route to the ComparIA ELO path."""
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
    assert capture_mains["args"].elo.arena == "ComparIA"


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


def test_pairwise_task_without_native_baseline_requires_model_a_and_b(capture_mains):
    with pytest.raises(SystemExit, match="--model_A and --model_B are required"):
        cli_module.cli(
            [
                "--task",
                "fluency-french",
                "--model_A",
                "Dummy/A",
                "--judge",
                "Dummy/J",
            ]
        )


@pytest.mark.parametrize(
    "task",
    [
        "alpaca-eval",
        "arena-hard-v0.1",
        "m-arena-hard-v2.0-EU",
        "mt-bench",
    ],
)
def test_pairwise_task_allows_missing_model_b_when_native_baseline_exists(
    capture_mains, task: str
):
    cli_module.cli(
        [
            "--task",
            task,
            "--model_A",
            "Dummy/A",
            "--judge",
            "Dummy/J",
        ]
    )
    assert capture_mains["module"] == "generate_and_evaluate"
    ge_cfg: RunConfig = capture_mains["args"]
    assert ge_cfg.task == task
    assert ge_cfg.model.path == "Dummy/A"
    assert ge_cfg.model.path_b is None


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
    elo_cfg: RunConfig = capture_mains["args"]
    assert elo_cfg.elo is not None
    assert elo_cfg.elo.languages == ["en", "fr"]
    assert elo_cfg.elo.n_instructions_per_language == 50
    assert elo_cfg.elo.n_bootstraps == 5
    assert elo_cfg.run.seed == 7
    assert elo_cfg.elo.baseline_model == "gpt-4o"


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
    ge_cfg: RunConfig = capture_mains["args"]
    assert ge_cfg.model.engine_kwargs == {"tensor_parallel_size": 4}


def test_missing_judge_model_errors(monkeypatch):
    monkeypatch.setattr(cli_module, "configure_logging", lambda *a, **k: None)
    with pytest.raises(SystemExit):
        cli_module.cli(
            ["--task", "alpaca-eval", "--model_A", "Dummy/a", "--model_B", "Dummy/b"]
        )


def test_judge_side_kwargs_are_parsed_separately(capture_mains):
    cli_module.cli(
        [
            "--task",
            "arena-hard-v2.0",
            "--model_A",
            "Dummy/A",
            "--judge",
            "Dummy/J",
            "--truncate_judge_input_chars",
            "80000",
            "--max_model_len",
            "32768",
            "--max_judge_model_len",
            "65536",
            "--engine_kwargs",
            '{"tensor_parallel_size": 1}',
            "--judge_engine_kwargs",
            '{"tensor_parallel_size": 4}',
        ]
    )
    ge_cfg: RunConfig = capture_mains["args"]
    assert ge_cfg.model.path_b is None
    assert ge_cfg.generation.truncate_judge_input_chars == 80000
    assert ge_cfg.model.max_model_len == 32768
    assert ge_cfg.judge.max_model_len == 65536
    assert ge_cfg.model.engine_kwargs == {"tensor_parallel_size": 1}
    assert ge_cfg.judge.engine_kwargs == {"tensor_parallel_size": 4}
