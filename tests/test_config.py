from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import judgearena.config as config_module
from judgearena.config import RunConfig, build_run_config, dump_config, load_config


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


def test_load_config_ignores_unused_legacy_judge_fields(tmp_path):
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        "task: alpaca-eval\n"
        "model: {name: Dummy/a, baseline: Dummy/b}\n"
        "judge:\n"
        "  model: Dummy/j\n"
        "  provide_explanation: false\n"
        "  system_prompt_file: null\n"
        "  user_prompt_file: null\n"
    )

    assert load_config(yaml_path) == RunConfig(**_base_generate())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provide_explanation", True),
        ("system_prompt_file", "system.txt"),
        ("prompt_presett", "default"),
    ],
)
def test_active_legacy_and_unknown_judge_fields_are_rejected(field, value):
    data = _base_generate()
    data["judge"][field] = value

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunConfig(**data)


def _registered_task():
    judge = SimpleNamespace(
        default_swap_mode="both",
        default_temperature=0.25,
        default_max_out_tokens=4096,
        default_top_logprobs=5,
    )
    return SimpleNamespace(spec=SimpleNamespace(protocol=SimpleNamespace(judge=judge)))


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, ("both", 0.25, 4096, 5)),
        (
            {
                "swap_mode": "random",
                "temperature": None,
                "max_out_tokens": 512,
                "top_logprobs": 2,
            },
            ("random", None, 512, 2),
        ),
    ],
)
def test_registered_task_defaults_do_not_replace_explicit_judge_config(
    monkeypatch, overrides, expected
):
    monkeypatch.setattr(
        config_module, "get_packaged_task", lambda _task: _registered_task()
    )
    data = _base_generate()
    data["task"] = "yaml-task"
    data["judge"].update(overrides)

    cfg = RunConfig(**data)

    assert (
        cfg.judge.swap_mode,
        cfg.judge.temperature,
        cfg.judge.max_out_tokens,
        cfg.judge.top_logprobs,
    ) == expected


def test_elo_config_derives_scoring_defaults():
    cfg = RunConfig(**_base_elo())
    assert cfg.elo is not None
    assert cfg.elo.soft_elo is True
    assert cfg.elo.soft_elo_temperature == 0.3


def test_elo_config_allows_runtime_scoring_overrides():
    data = _base_elo()
    data["elo"] = {"soft_elo": False, "soft_elo_temperature": 0.7}

    cfg = RunConfig(**data)

    assert cfg.elo is not None
    assert cfg.elo.soft_elo is False
    assert cfg.elo.soft_elo_temperature == 0.7


def test_elo_block_rejected_on_generate_task():
    data = _base_generate()
    data["elo"] = {"n_bootstraps": 5}
    with pytest.raises(ValidationError):
        RunConfig(**data)


def test_generate_requires_model_path():
    data = _base_generate()
    data["model"] = {"baseline": "Dummy/b"}  # no path
    with pytest.raises(ValidationError):
        RunConfig(**data)


def test_dump_config_round_trips_custom_prompt_paths(tmp_path):
    system_file = tmp_path / "system.txt"
    user_file = tmp_path / "user.txt"
    system_file.write_text("Judge carefully.")
    user_file.write_text(
        "Instruction: {user_prompt}\nA: {completion_A}\nB: {completion_B}"
    )
    cfg = RunConfig(
        task="alpaca-eval",
        model={"name": "Dummy/a", "baseline": "Dummy/b"},
        judge={
            "model": "Dummy/j",
            "prompt": {
                "system_file": system_file,
                "user_file": user_file,
                "parser": "score",
            },
        },
    )
    config_path = tmp_path / "resolved.yaml"

    dump_config(cfg, config_path)

    assert load_config(config_path) == cfg


def test_cli_yaml_equivalence_generate(tmp_path):
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


def test_build_run_config_cli_overrides_yaml_partial(tmp_path):
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
