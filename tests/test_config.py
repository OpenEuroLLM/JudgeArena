import pytest
from pydantic import ValidationError

from judgearena import cli as cli_module
from judgearena.config import RunConfig


def _base_generate() -> dict:
    return {
        "task": "alpaca-eval",
        "model": {"path": "Dummy/a", "path_b": "Dummy/b"},
        "judge": {"model": "Dummy/j"},
    }


def _base_elo() -> dict:
    return {
        "task": "elo-comparia",
        "model": {"path": "Dummy/m"},
        "judge": {"model": "Dummy/j"},
    }


def test_generate_config_constructs():
    cfg = RunConfig(**_base_generate())
    assert cfg.task == "alpaca-eval"
    assert cfg.model.path == "Dummy/a"
    assert cfg.judge.model == "Dummy/j"
    assert cfg.elo is None


def test_elo_config_derives_arena():
    cfg = RunConfig(**_base_elo())
    assert cfg.elo is not None
    assert cfg.elo.arena == "ComparIA"


def test_elo_requires_model_path():
    data = _base_elo()
    data["model"] = {}
    with pytest.raises(ValidationError):
        RunConfig(**data)


def test_elo_block_rejected_on_generate_task():
    data = _base_generate()
    data["elo"] = {"n_bootstraps": 5}
    with pytest.raises(ValidationError):
        RunConfig(**data)


def test_generate_requires_model_b_without_native_baseline():
    data = _base_generate()
    data["task"] = "no-baseline-task"  # task with no native baseline
    data["model"] = {"path": "Dummy/a"}  # no path_b
    with pytest.raises(ValidationError):
        RunConfig(**data)


def test_unknown_elo_task_rejected():
    data = _base_elo()
    data["task"] = "elo-nope"
    with pytest.raises(ValidationError):
        RunConfig(**data)


def test_generate_requires_model_path():
    data = _base_generate()
    data["model"] = {"path_b": "Dummy/b"}  # no path
    with pytest.raises(ValidationError):
        RunConfig(**data)


def test_load_config_from_yaml(tmp_path):
    from judgearena.config import load_config

    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        "task: alpaca-eval\n"
        "model:\n"
        "  path: Dummy/a\n"
        "  path_b: Dummy/b\n"
        "  max_out_tokens: 4096\n"
        "judge:\n"
        "  model: Dummy/j\n"
        "  provide_explanation: true\n"
        "generation:\n"
        "  n_instructions: 10\n"
    )
    cfg = load_config(yaml_path)
    assert cfg.model.path == "Dummy/a"
    assert cfg.model.max_out_tokens == 4096
    assert cfg.judge.provide_explanation is True
    assert cfg.generation.n_instructions == 10


def test_cli_yaml_equivalence_generate(tmp_path):
    from judgearena.config import build_run_config, load_config

    expected = build_run_config(
        ["--task", "alpaca-eval", "--model.path", "Dummy/a",
         "--model.path_b", "Dummy/b", "--judge.model", "Dummy/j"]
    )
    yaml_path = tmp_path / "g.yaml"
    yaml_path.write_text(
        "task: alpaca-eval\n"
        "model: {path: Dummy/a, path_b: Dummy/b}\n"
        "judge: {model: Dummy/j}\n"
    )
    actual = load_config(yaml_path)
    assert actual == expected


def test_cli_yaml_equivalence_elo(tmp_path):
    from judgearena.config import build_run_config, load_config

    expected = build_run_config(
        ["--task", "elo-comparia", "--model.path", "Dummy/m", "--judge.model", "Dummy/j"]
    )
    yaml_path = tmp_path / "e.yaml"
    yaml_path.write_text(
        "task: elo-comparia\n"
        "model: {path: Dummy/m}\n"
        "judge: {model: Dummy/j}\n"
    )
    actual = load_config(yaml_path)
    assert actual == expected


def test_config_path_dispatches_elo(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "main_elo", lambda a: captured.setdefault("elo", a))
    monkeypatch.setattr(
        cli_module,
        "main_generate_and_evaluate",
        lambda a: captured.setdefault("ge", a),
    )
    yaml_path = tmp_path / "e.yaml"
    yaml_path.write_text(
        "task: elo-comparia\nmodel: {path: Dummy/m}\njudge: {model: Dummy/j}\n"
    )
    cli_module.cli(["--config_path", str(yaml_path)])
    assert "ge" not in captured
    assert isinstance(captured["elo"], RunConfig)
    assert captured["elo"].elo is not None
    assert captured["elo"].elo.arena == "ComparIA"


def test_build_run_config_cli_only():
    from judgearena.config import build_run_config

    cfg = build_run_config(
        ["--task", "alpaca-eval", "--model.path", "Dummy/a",
         "--model.path_b", "Dummy/b", "--judge.model", "Dummy/j"]
    )
    assert cfg.task == "alpaca-eval"
    assert cfg.model.path == "Dummy/a"
    assert cfg.model.path_b == "Dummy/b"
    assert cfg.judge.model == "Dummy/j"


def test_build_run_config_cli_overrides_yaml_partial(tmp_path):
    from judgearena.config import build_run_config

    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        "task: alpaca-eval\n"
        "model: {path: Dummy/a, path_b: Dummy/b}\n"
        "judge: {model: yaml-judge, swap_mode: both}\n"
    )
    cfg = build_run_config(
        ["--config_path", str(yaml_path), "--judge.model", "cli-judge"]
    )
    assert cfg.judge.model == "cli-judge"  # CLI overrides YAML
    assert cfg.judge.swap_mode == "both"  # preserved (partial update)
    assert cfg.model.path == "Dummy/a"  # from YAML
    assert cfg.generation.truncate_all_input_chars == 8192  # model default


def test_build_run_config_engine_kwargs_json():
    from judgearena.config import build_run_config

    cfg = build_run_config(
        ["--task", "alpaca-eval", "--model.path", "Dummy/a", "--model.path_b", "Dummy/b",
         "--judge.model", "Dummy/j", "--judge.engine_kwargs", '{"tensor_parallel_size": 4}']
    )
    assert cfg.judge.engine_kwargs == {"tensor_parallel_size": 4}


def test_build_run_config_elo_arena_derived():
    from judgearena.config import build_run_config

    cfg = build_run_config(
        ["--task", "elo-comparia", "--model.path", "Dummy/m", "--judge.model", "Dummy/j"]
    )
    assert cfg.elo is not None and cfg.elo.arena == "ComparIA"
