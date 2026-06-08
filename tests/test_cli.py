"""Tests for the unified `judgearena` CLI dispatcher (model-driven CLI)."""

from __future__ import annotations

import pytest

from judgearena import cli as cli_module
from judgearena.config import RunConfig


@pytest.fixture
def capture_mains(monkeypatch):
    """Replace both main functions (and logging) with spies recording the config."""
    captured: dict[str, object] = {}

    def fake_main_ge(cfg: RunConfig) -> None:
        captured["module"] = "generate_and_evaluate"
        captured["cfg"] = cfg

    def fake_main_elo(cfg: RunConfig) -> None:
        captured["module"] = "elo"
        captured["cfg"] = cfg

    monkeypatch.setattr(cli_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "main_generate_and_evaluate", fake_main_ge)
    monkeypatch.setattr(cli_module, "main_elo", fake_main_elo)
    return captured


@pytest.mark.parametrize(
    "task",
    ["alpaca-eval", "arena-hard-v2.0", "m-arena-hard-v2.0-EU", "fluency-french", "mt-bench"],
)
def test_task_dispatches_to_generate_and_evaluate(capture_mains, task: str):
    cli_module.cli(
        ["--task", task, "--model.path", "Dummy/A",
         "--model.path_b", "Dummy/B", "--judge.model", "Dummy/J"]
    )
    assert capture_mains["module"] == "generate_and_evaluate"
    cfg = capture_mains["cfg"]
    assert isinstance(cfg, RunConfig)
    assert cfg.task == task
    assert cfg.model.path == "Dummy/A"
    assert cfg.model.path_b == "Dummy/B"
    assert cfg.judge.model == "Dummy/J"


@pytest.mark.parametrize(
    "task, expected_arena",
    [
        ("elo-comparia", "ComparIA"),
        ("elo-lmarena-140k", "LMArena-140k"),
        ("elo-lmarena-100k", "LMArena-100k"),
        ("elo-lmarena", "LMArena"),
    ],
)
def test_elo_task_dispatches(capture_mains, task: str, expected_arena: str):
    cli_module.cli(
        ["--task", task, "--model.path", "Dummy/X", "--judge.model", "Dummy/J"]
    )
    assert capture_mains["module"] == "elo"
    cfg = capture_mains["cfg"]
    assert isinstance(cfg, RunConfig)
    assert cfg.elo is not None
    assert cfg.elo.arena == expected_arena
    assert cfg.model.path == "Dummy/X"


def test_config_path_dispatches_and_cli_overrides(tmp_path, capture_mains):
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        "task: alpaca-eval\n"
        "model: {path: Dummy/A, path_b: Dummy/B}\n"
        "judge: {model: yaml-judge, swap_mode: both}\n"
    )
    cli_module.cli(["--config_path", str(yaml_path), "--judge.model", "cli-judge"])
    cfg = capture_mains["cfg"]
    assert cfg.judge.model == "cli-judge"  # CLI overrides YAML
    assert cfg.judge.swap_mode == "both"  # preserved from YAML


def test_missing_task_errors(capture_mains):
    with pytest.raises(SystemExit):
        cli_module.cli(["--model.path", "Dummy/A", "--judge.model", "Dummy/J"])


def test_missing_judge_errors(capture_mains):
    with pytest.raises(SystemExit):
        cli_module.cli(["--task", "alpaca-eval", "--model.path", "Dummy/A",
                        "--model.path_b", "Dummy/B"])


def test_elo_requires_model_path(capture_mains):
    with pytest.raises(SystemExit, match="model.path is required for elo"):
        cli_module.cli(["--task", "elo-comparia", "--judge.model", "Dummy/J"])


def test_elo_rejects_model_path_b(capture_mains):
    with pytest.raises(SystemExit, match="model.path_b is not supported for elo"):
        cli_module.cli(["--task", "elo-comparia", "--model.path", "Dummy/X",
                        "--model.path_b", "Dummy/B", "--judge.model", "Dummy/J"])


def test_unknown_elo_task_errors(capture_mains):
    with pytest.raises(SystemExit, match="Unknown elo task"):
        cli_module.cli(["--task", "elo-foo", "--model.path", "Dummy/X",
                        "--judge.model", "Dummy/J"])


def test_pairwise_without_native_baseline_requires_model_path_b(capture_mains):
    with pytest.raises(SystemExit, match="model.path_b is required"):
        cli_module.cli(["--task", "fluency-french", "--model.path", "Dummy/A",
                        "--judge.model", "Dummy/J"])


@pytest.mark.parametrize(
    "task", ["alpaca-eval", "arena-hard-v0.1", "m-arena-hard-v2.0-EU", "mt-bench"]
)
def test_pairwise_allows_missing_model_path_b_with_native_baseline(capture_mains, task):
    cli_module.cli(["--task", task, "--model.path", "Dummy/A", "--judge.model", "Dummy/J"])
    cfg = capture_mains["cfg"]
    assert cfg.task == task
    assert cfg.model.path_b is None


def test_elo_forwards_optional_flags(capture_mains):
    cli_module.cli(
        ["--task", "elo-lmarena-140k", "--model.path", "Dummy/X", "--judge.model", "Dummy/J",
         "--elo.languages", '["en", "fr"]', "--elo.n_instructions_per_language", "50",
         "--elo.n_bootstraps", "5", "--run.seed", "7", "--elo.baseline_model", "gpt-4o"]
    )
    cfg = capture_mains["cfg"]
    assert cfg.elo is not None
    assert cfg.elo.languages == ["en", "fr"]
    assert cfg.elo.n_instructions_per_language == 50
    assert cfg.elo.n_bootstraps == 5
    assert cfg.run.seed == 7
    assert cfg.elo.baseline_model == "gpt-4o"


def test_engine_kwargs_parsed_as_json(capture_mains):
    cli_module.cli(
        ["--task", "alpaca-eval", "--model.path", "Dummy/A", "--model.path_b", "Dummy/B",
         "--judge.model", "Dummy/J", "--model.engine_kwargs", '{"tensor_parallel_size": 4}']
    )
    cfg = capture_mains["cfg"]
    assert cfg.model.engine_kwargs == {"tensor_parallel_size": 4}


def test_judge_side_kwargs_parsed_separately(capture_mains):
    cli_module.cli(
        ["--task", "arena-hard-v2.0", "--model.path", "Dummy/A", "--judge.model", "Dummy/J",
         "--generation.truncate_judge_input_chars", "80000",
         "--model.max_model_len", "32768", "--judge.max_model_len", "65536",
         "--model.engine_kwargs", '{"tensor_parallel_size": 1}',
         "--judge.engine_kwargs", '{"tensor_parallel_size": 4}']
    )
    cfg = capture_mains["cfg"]
    assert cfg.model.path_b is None
    assert cfg.generation.truncate_judge_input_chars == 80000
    assert cfg.model.max_model_len == 32768
    assert cfg.judge.max_model_len == 65536
    assert cfg.model.engine_kwargs == {"tensor_parallel_size": 1}
    assert cfg.judge.engine_kwargs == {"tensor_parallel_size": 4}
