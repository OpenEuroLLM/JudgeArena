from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import judgearena.config as config_module
from judgearena import cli as cli_module
from judgearena.config import RunConfig


def _base_generate() -> dict:
    return {
        "task": "alpaca-eval",
        "model": {"name": "Dummy/a", "baseline": "Dummy/b"},
        "judge": {"model": "Dummy/j"},
    }


def _base_elo() -> dict:
    return {
        "task": "elo-comparia",
        "model": {"name": "Dummy/m"},
        "judge": {"model": "Dummy/j"},
    }


def test_generate_config_constructs():
    cfg = RunConfig(**_base_generate())
    assert cfg.task == "alpaca-eval"
    assert cfg.model.name == "Dummy/a"
    assert cfg.judge.model == "Dummy/j"
    assert cfg.elo is None


def _registered_task(
    *,
    default_swap_mode: str = "both",
    allowed_swap_modes: tuple[str, ...] = ("both",),
    default_temperature: float | None = 0.25,
    default_max_out_tokens: int | None = 4096,
    default_top_logprobs: int | None = 5,
    allow_runtime_override: bool = True,
):
    return SimpleNamespace(
        spec=SimpleNamespace(
            protocol=SimpleNamespace(
                judge=SimpleNamespace(
                    default_swap_mode=default_swap_mode,
                    allowed_swap_modes=allowed_swap_modes,
                    default_temperature=default_temperature,
                    default_max_out_tokens=default_max_out_tokens,
                    default_top_logprobs=default_top_logprobs,
                ),
                baseline=SimpleNamespace(allow_runtime_override=allow_runtime_override),
            )
        )
    )


def test_registered_task_applies_judge_defaults(monkeypatch):
    monkeypatch.setattr(
        config_module, "get_packaged_task", lambda _task: _registered_task()
    )
    data = _base_generate()
    data["task"] = "yaml-task"

    cfg = RunConfig(**data)

    assert cfg.judge.swap_mode == "both"
    assert cfg.judge.temperature == 0.25
    assert cfg.judge.max_out_tokens == 4096
    assert cfg.judge.top_logprobs == 5


def test_registered_task_keeps_explicit_judge_max_out_tokens(monkeypatch):
    monkeypatch.setattr(
        config_module, "get_packaged_task", lambda _task: _registered_task()
    )
    data = _base_generate()
    data["task"] = "yaml-task"
    data["judge"]["max_out_tokens"] = 512

    cfg = RunConfig(**data)

    assert cfg.judge.max_out_tokens == 512


def test_registered_task_rejects_unsupported_swap_mode(monkeypatch):
    monkeypatch.setattr(
        config_module, "get_packaged_task", lambda _task: _registered_task()
    )
    data = _base_generate()
    data["task"] = "yaml-task"
    data["judge"]["swap_mode"] = "fixed"

    with pytest.raises(ValidationError, match="not supported"):
        RunConfig(**data)


def test_registered_task_can_forbid_baseline_override(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "get_packaged_task",
        lambda _task: _registered_task(allow_runtime_override=False),
    )
    data = _base_generate()
    data["task"] = "yaml-task"

    with pytest.raises(ValidationError, match="cannot override"):
        RunConfig(**data)


def test_elo_config_derives_arena():
    cfg = RunConfig(**_base_elo())
    assert cfg.elo is not None
    assert cfg.elo.arena == "ComparIA"
    assert cfg.elo.soft_elo is True
    assert cfg.elo.soft_elo_temperature == 0.3


def test_elo_config_rejects_arena_that_conflicts_with_task():
    data = _base_elo()
    data["elo"] = {"arena": "LMArena-100k"}

    with pytest.raises(ValidationError, match="does not match task"):
        RunConfig(**data)


def test_elo_config_allows_runtime_scoring_overrides():
    data = _base_elo()
    data["elo"] = {"soft_elo": False, "soft_elo_temperature": 0.7}

    cfg = RunConfig(**data)

    assert cfg.elo is not None
    assert cfg.elo.soft_elo is False
    assert cfg.elo.soft_elo_temperature == 0.7


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
    data["model"] = {"name": "Dummy/a"}  # no path_b
    with pytest.raises(ValidationError):
        RunConfig(**data)


def test_unknown_elo_task_rejected():
    data = _base_elo()
    data["task"] = "elo-nope"
    with pytest.raises(ValidationError):
        RunConfig(**data)


def test_generate_requires_model_path():
    data = _base_generate()
    data["model"] = {"baseline": "Dummy/b"}  # no path
    with pytest.raises(ValidationError):
        RunConfig(**data)


def test_load_config_from_yaml(tmp_path):
    from judgearena.config import load_config

    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        "task: alpaca-eval\n"
        "model:\n"
        "  name: Dummy/a\n"
        "  baseline: Dummy/b\n"
        "  max_out_tokens: 4096\n"
        "judge:\n"
        "  model: Dummy/j\n"
        "  provide_explanation: true\n"
        "generation:\n"
        "  n_instructions: 10\n"
    )
    cfg = load_config(yaml_path)
    assert cfg.model.name == "Dummy/a"
    assert cfg.model.max_out_tokens == 4096
    assert cfg.judge.provide_explanation is True
    assert cfg.generation.n_instructions == 10


def test_cli_yaml_equivalence_generate(tmp_path):
    from judgearena.config import build_run_config, load_config

    expected = build_run_config(
        [
            "--task",
            "alpaca-eval",
            "--model.name",
            "Dummy/a",
            "--model.baseline",
            "Dummy/b",
            "--judge.model",
            "Dummy/j",
        ]
    )
    yaml_path = tmp_path / "g.yaml"
    yaml_path.write_text(
        "task: alpaca-eval\n"
        "model: {name: Dummy/a, baseline: Dummy/b}\n"
        "judge: {model: Dummy/j}\n"
    )
    actual = load_config(yaml_path)
    assert actual == expected


def test_cli_yaml_equivalence_elo(tmp_path):
    from judgearena.config import build_run_config, load_config

    expected = build_run_config(
        [
            "--task",
            "elo-comparia",
            "--model.name",
            "Dummy/m",
            "--judge.model",
            "Dummy/j",
        ]
    )
    yaml_path = tmp_path / "e.yaml"
    yaml_path.write_text(
        "task: elo-comparia\nmodel: {name: Dummy/m}\njudge: {model: Dummy/j}\n"
    )
    actual = load_config(yaml_path)
    assert actual == expected


def test_config_path_dispatches_elo(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_module, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        cli_module,
        "run_benchmark",
        lambda a: captured.setdefault("benchmark", a),
    )
    yaml_path = tmp_path / "e.yaml"
    yaml_path.write_text(
        "task: elo-comparia\nmodel: {name: Dummy/m}\njudge: {model: Dummy/j}\n"
    )
    cli_module.cli(["--config_path", str(yaml_path)])
    assert isinstance(captured["benchmark"], RunConfig)
    assert captured["benchmark"].elo is not None
    assert captured["benchmark"].elo.arena == "ComparIA"


def test_build_run_config_cli_only():
    from judgearena.config import build_run_config

    cfg = build_run_config(
        [
            "--task",
            "alpaca-eval",
            "--model.name",
            "Dummy/a",
            "--model.baseline",
            "Dummy/b",
            "--judge.model",
            "Dummy/j",
        ]
    )
    assert cfg.task == "alpaca-eval"
    assert cfg.model.name == "Dummy/a"
    assert cfg.model.baseline == "Dummy/b"
    assert cfg.judge.model == "Dummy/j"


def test_build_run_config_cli_overrides_yaml_partial(tmp_path):
    from judgearena.config import build_run_config

    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        "task: alpaca-eval\n"
        "model: {name: Dummy/a, baseline: Dummy/b}\n"
        "judge: {model: yaml-judge, swap_mode: both}\n"
    )
    cfg = build_run_config(
        ["--config_path", str(yaml_path), "--judge.model", "cli-judge"]
    )
    assert cfg.judge.model == "cli-judge"  # CLI overrides YAML
    assert cfg.judge.swap_mode == "both"  # preserved (partial update)
    assert cfg.model.name == "Dummy/a"  # from YAML
    assert cfg.generation.truncate_all_input_chars == 8192  # model default


def test_build_run_config_engine_kwargs_json():
    from judgearena.config import build_run_config

    cfg = build_run_config(
        [
            "--task",
            "alpaca-eval",
            "--model.name",
            "Dummy/a",
            "--model.baseline",
            "Dummy/b",
            "--judge.model",
            "Dummy/j",
            "--judge.engine_kwargs",
            '{"tensor_parallel_size": 4}',
        ]
    )
    assert cfg.judge.engine_kwargs == {"tensor_parallel_size": 4}


def test_build_run_config_elo_arena_derived():
    from judgearena.config import build_run_config

    cfg = build_run_config(
        [
            "--task",
            "elo-comparia",
            "--model.name",
            "Dummy/m",
            "--judge.model",
            "Dummy/j",
        ]
    )
    assert cfg.elo is not None and cfg.elo.arena == "ComparIA"
