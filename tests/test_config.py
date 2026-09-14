from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import judgearena.config as config_module
from judgearena import cli as cli_module
from judgearena.benchmarks import execution as execution_module
from judgearena.config import EloArgs, RunConfig, dump_config, load_config
from judgearena.tasks.schema import EloScoringSpec, MetricSpec


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
        ("user_prompt_file", "user.txt"),
        ("prompt_presett", "default"),
    ],
)
def test_active_legacy_and_unknown_judge_fields_are_rejected(field, value):
    data = _base_generate()
    data["judge"][field] = value

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunConfig(**data)


def _registered_task(
    *,
    default_swap_mode: str = "both",
    default_temperature: float | None = 0.25,
    default_max_out_tokens: int | None = 4096,
    default_top_logprobs: int | None = 5,
):
    return SimpleNamespace(
        spec=SimpleNamespace(
            protocol=SimpleNamespace(
                generation=SimpleNamespace(default_max_out_tokens=1024, default_seed=7),
                judge=SimpleNamespace(
                    default_swap_mode=default_swap_mode,
                    default_temperature=default_temperature,
                    default_max_out_tokens=default_max_out_tokens,
                    default_top_logprobs=default_top_logprobs,
                ),
            )
        )
    )


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


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, (1024, 7, 1024, 7)),
        (
            {
                "engine_kwargs": {"max_tokens": "64", "seed": 0},
                "baseline_engine_kwargs": {"max_tokens": "32", "seed": 1},
                "baseline_seed": 2,
            },
            (64, 0, 32, 2),
        ),
        (
            {
                "max_out_tokens": 128,
                "seed": 0,
                "engine_kwargs": {"max_tokens": 64, "seed": 1},
            },
            (128, 0, 128, 0),
        ),
        ({"seed": None, "baseline_seed": None}, (1024, None, 1024, None)),
    ],
)
def test_generation_settings_prefer_dedicated_then_engine_then_task(
    monkeypatch, overrides, expected
):
    monkeypatch.setattr(
        config_module, "get_packaged_task", lambda _task: _registered_task()
    )
    data = _base_generate()
    data["model"].update(overrides)
    cfg = RunConfig(**data)
    model = cfg.model.evaluated_generation_kwargs()
    baseline = cfg.model.baseline_generation_kwargs()

    assert (
        model["max_tokens"],
        model.get("seed"),
        baseline["max_tokens"],
        baseline.get("seed"),
    ) == expected
    assert type(model["max_tokens"]) is int


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"engine_kwargs": {"max_tokens": "64"}}, (0.5, 64, 3)),
        (
            {
                "temperature": 0.0,
                "top_logprobs": 0,
                "engine_kwargs": {"temperature": 0.7, "top_logprobs": 2},
            },
            (0.0, 128, 0),
        ),
    ],
)
def test_judge_settings_include_inherited_engine_kwargs(
    monkeypatch, overrides, expected
):
    monkeypatch.setattr(
        config_module, "get_packaged_task", lambda _task: _registered_task()
    )
    monkeypatch.setattr(execution_module, "make_model", lambda **kwargs: kwargs)
    data = _base_generate()
    data["model"]["engine_kwargs"] = {
        "temperature": 0.5,
        "max_tokens": 128,
        "top_logprobs": 3,
    }
    data["judge"].update(model="VLLM/j", **overrides)

    kwargs = execution_module.build_judge(RunConfig(**data))

    assert (
        kwargs["temperature"],
        kwargs["max_tokens"],
        kwargs["top_logprobs"],
    ) == expected
    assert type(kwargs["max_tokens"]) is int


@pytest.mark.parametrize(
    ("task", "judge_model", "expected"),
    [
        ("alpaca-eval", "OpenRouter/j", 1),
        ("alpaca-eval", "VLLM/j", 128),
        ("mt-bench", "OpenRouter/j", 128),
    ],
)
def test_judge_engine_inheritance_matches_the_runner(
    monkeypatch, task, judge_model, expected
):
    data = _base_generate()
    data["task"] = task
    data["model"]["engine_kwargs"] = {"max_tokens": 128}
    data["judge"]["model"] = judge_model
    monkeypatch.setattr(execution_module, "make_model", lambda **kwargs: kwargs)

    cfg = RunConfig(**data)
    if task == "mt-bench":
        kwargs = cfg.judge.model_kwargs(base_engine_kwargs=cfg.model.engine_kwargs)
    else:
        kwargs = execution_module.build_judge(cfg)

    assert cfg.judge.max_out_tokens == kwargs["max_tokens"] == expected


@pytest.mark.parametrize("role", ["model", "judge"])
def test_engine_max_tokens_must_be_a_valid_token_limit(role):
    data = _base_generate()
    data["task"] = "mt-bench"
    data[role]["engine_kwargs"] = {"max_tokens": None}

    with pytest.raises(ValidationError, match="max_out_tokens"):
        RunConfig(**data)


@pytest.mark.parametrize(
    ("task", "generation", "expected"),
    [
        ("arena-hard-v2.0", {}, None),
        ("arena-hard-v2.0-ja", {}, 8192),
        ("arena-hard-v2.0-ja", {"truncate_all_input_chars": None}, None),
    ],
)
def test_generation_truncation_defaults_preserve_explicit_overrides(
    task, generation, expected
):
    data = _base_generate()
    data.update(task=task, generation=generation)

    assert RunConfig(**data).generation.truncate_all_input_chars == expected


def test_elo_config_keeps_defaults_implicit_until_task_resolution():
    cfg = RunConfig(**_base_elo())
    assert cfg.elo is not None
    assert cfg.elo.soft_elo is True
    assert cfg.elo.soft_elo_temperature == 0.3
    assert cfg.elo.model_fields_set == set()


@pytest.mark.parametrize(
    ("parameters", "runtime", "expected"),
    [
        ({}, {}, (20, None, False, 0.7)),
        (
            {"n_bootstraps": 2, "baseline_model": "anchor", "soft": True},
            {},
            (2, "anchor", True, 0.7),
        ),
        (
            {"n_bootstraps": 2, "baseline_model": "anchor", "soft": True},
            {
                "n_bootstraps": 0,
                "baseline_model": None,
                "soft_elo": False,
                "soft_elo_temperature": 0.2,
            },
            (0, None, False, 0.2),
        ),
    ],
)
def test_elo_resolution_prefers_runtime_then_metric_then_defaults(
    parameters, runtime, expected
):
    scoring = EloScoringSpec(
        metrics=(MetricSpec(metric="bradley_terry", parameters=parameters),),
        default_soft=False,
        default_temperature=0.7,
    )

    resolved = EloArgs(**runtime).resolve(scoring)

    assert (
        resolved.n_bootstraps,
        resolved.baseline_model,
        resolved.soft_elo,
        resolved.soft_elo_temperature,
    ) == expected


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
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(
        "task: alpaca-eval\n"
        "model:\n"
        "  name: Dummy/a\n"
        "  baseline: Dummy/b\n"
        "  max_out_tokens: 4096\n"
        "judge:\n"
        "  model: Dummy/j\n"
        "  prompt_preset: default_with_explanation\n"
        "generation:\n"
        "  n_instructions: 10\n"
    )
    cfg = load_config(yaml_path)
    assert cfg.model.name == "Dummy/a"
    assert cfg.model.max_out_tokens == 4096
    assert cfg.judge.prompt_preset == "default_with_explanation"
    assert cfg.generation.n_instructions == 10


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


def test_build_run_config_elo_defaults():
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
    assert cfg.elo is not None
    assert cfg.elo.soft_elo is True
