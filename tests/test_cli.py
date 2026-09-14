"""Tests for the unified `judgearena` CLI dispatcher (model-driven CLI)."""

from __future__ import annotations

import pytest

from judgearena import cli as cli_module
from judgearena.config import RunConfig


@pytest.fixture
def capture_mains(monkeypatch):
    """Replace benchmark execution and logging with a config spy."""
    captured: dict[str, object] = {}

    def fake_run_benchmark(cfg: RunConfig) -> None:
        captured["cfg"] = cfg

    monkeypatch.setattr(cli_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "run_benchmark", fake_run_benchmark)
    return captured


def test_pairwise_dispatches_with_native_baseline(capture_mains):
    cli_module.cli(
        ["--task", "alpaca-eval", "--model.name", "Dummy/A", "--judge.model", "Dummy/J"]
    )
    cfg = capture_mains["cfg"]
    assert isinstance(cfg, RunConfig)
    assert cfg.task == "alpaca-eval"
    assert cfg.model.baseline is None


def test_config_path_dispatches(tmp_path, capture_mains):
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        "task: alpaca-eval\n"
        "model: {name: Dummy/A, baseline: Dummy/B}\n"
        "judge: {model: yaml-judge, swap_mode: both}\n"
    )
    cli_module.cli(["--config_path", str(yaml_path)])
    cfg = capture_mains["cfg"]
    assert cfg.judge.model == "yaml-judge"


def test_missing_task_errors(capture_mains):
    with pytest.raises(SystemExit):
        cli_module.cli(["--model.name", "Dummy/A", "--judge.model", "Dummy/J"])


def test_elo_requires_model_path(capture_mains):
    with pytest.raises(SystemExit, match="model.name is required for ELO"):
        cli_module.cli(["--task", "elo-comparia", "--judge.model", "Dummy/J"])


def test_elo_rejects_model_path_b(capture_mains):
    with pytest.raises(SystemExit, match="model.baseline is not supported for ELO"):
        cli_module.cli(
            [
                "--task",
                "elo-comparia",
                "--model.name",
                "Dummy/X",
                "--model.baseline",
                "Dummy/B",
                "--judge.model",
                "Dummy/J",
            ]
        )


def test_unknown_elo_task_errors(capture_mains):
    with pytest.raises(SystemExit, match="Unknown task"):
        cli_module.cli(
            ["--task", "elo-foo", "--model.name", "Dummy/X", "--judge.model", "Dummy/J"]
        )


def test_pairwise_without_native_baseline_requires_model_path_b(capture_mains):
    with pytest.raises(SystemExit, match="model.baseline is required"):
        cli_module.cli(
            [
                "--task",
                "fluency-french",
                "--model.name",
                "Dummy/A",
                "--judge.model",
                "Dummy/J",
            ]
        )


def test_elo_dispatches_with_optional_flags(capture_mains):
    cli_module.cli(
        [
            "--task",
            "elo-lmarena-140k",
            "--model.name",
            "Dummy/X",
            "--judge.model",
            "Dummy/J",
            "--elo.languages",
            '["en", "fr"]',
            "--elo.n_bootstraps",
            "5",
            "--judge.prompt_preset",
            "default_with_explanation",
        ]
    )
    cfg = capture_mains["cfg"]
    assert cfg.elo.languages == ["en", "fr"]
    assert cfg.elo.n_bootstraps == 5
    assert cfg.judge.prompt_preset == "default_with_explanation"


def test_custom_judge_prompt_files_are_forwarded(capture_mains, tmp_path):
    system_file = tmp_path / "system.txt"
    user_file = tmp_path / "user.txt"
    system_file.write_text("system prompt")
    user_file.write_text("user template")
    cli_module.cli(
        [
            "--task",
            "alpaca-eval",
            "--model.name",
            "Dummy/A",
            "--model.baseline",
            "Dummy/B",
            "--judge.model",
            "Dummy/J",
            "--judge.prompt.system_file",
            str(system_file),
            "--judge.prompt.user_file",
            str(user_file),
        ]
    )
    cfg = capture_mains["cfg"]
    assert cfg.judge.prompt.system_file == system_file
    assert cfg.judge.prompt.user_file == user_file
    assert cfg.judge.prompt.parser == "score"


def test_judge_side_kwargs_parsed_separately(capture_mains):
    cli_module.cli(
        [
            "--task",
            "arena-hard-v2.0",
            "--model.name",
            "Dummy/A",
            "--judge.model",
            "Dummy/J",
            "--generation.truncate_judge_input_chars",
            "80000",
            "--model.max_model_len",
            "32768",
            "--judge.max_model_len",
            "65536",
            "--model.engine_kwargs",
            '{"tensor_parallel_size": 1}',
            "--judge.engine_kwargs",
            '{"tensor_parallel_size": 4}',
        ]
    )
    cfg = capture_mains["cfg"]
    assert cfg.model.baseline is None
    assert cfg.generation.truncate_judge_input_chars == 80000
    assert cfg.model.max_model_len == 32768
    assert cfg.judge.max_model_len == 65536
    assert cfg.model.engine_kwargs == {"tensor_parallel_size": 1}
    assert cfg.judge.engine_kwargs == {"tensor_parallel_size": 4}
