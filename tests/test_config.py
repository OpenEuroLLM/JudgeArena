import pytest
from pydantic import ValidationError

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


def _base_meta_eval() -> dict:
    return {
        "task": "meta-eval",
        "judge": {"model": "Dummy/j"},
    }


def test_generate_config_constructs():
    cfg = RunConfig(**_base_generate())
    assert cfg.task == "alpaca-eval"
    assert cfg.model.name == "Dummy/a"
    assert cfg.judge.model == "Dummy/j"
    assert cfg.elo is None


def test_elo_config_derives_arena():
    cfg = RunConfig(**_base_elo())
    assert cfg.elo is not None
    assert cfg.elo.arena == "ComparIA"


def test_meta_eval_config_constructs():
    cfg = RunConfig(**_base_meta_eval())
    assert cfg.meta_eval is not None
    assert cfg.meta_eval.reference_arena == "LMArena-140k"
    assert cfg.model.name is None


def test_meta_eval_rejects_model_config():
    data = _base_meta_eval()
    data["model"] = {"name": "Dummy/a"}
    with pytest.raises(ValidationError, match="model config is not used"):
        RunConfig(**data)


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
    monkeypatch.setattr(cli_module, "main_elo", lambda a: captured.setdefault("elo", a))
    monkeypatch.setattr(
        cli_module,
        "main_generate_and_evaluate",
        lambda a: captured.setdefault("ge", a),
    )
    yaml_path = tmp_path / "e.yaml"
    yaml_path.write_text(
        "task: elo-comparia\nmodel: {name: Dummy/m}\njudge: {model: Dummy/j}\n"
    )
    cli_module.cli(["--config_path", str(yaml_path)])
    assert "ge" not in captured
    assert isinstance(captured["elo"], RunConfig)
    assert captured["elo"].elo is not None
    assert captured["elo"].elo.arena == "ComparIA"


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


def test_cache_defaults():
    cfg = RunConfig(**_base_generate())
    assert cfg.cache.store_root is None
    assert cfg.cache.cache_mode == "use"
    assert cfg.cache.cache_hf_repo == "judge-arena/judge-arena-cache"
    assert cfg.cache.cache_fetch is False
    assert cfg.cache.cache_push is False
    assert cfg.cache.cache_create_pr is False


def test_cache_pushed_by_defaults_to_getuser(monkeypatch):
    monkeypatch.setattr("judgearena.config.default_pushed_by", lambda: "test-user")
    cfg = RunConfig(**_base_generate())
    assert cfg.cache.pushed_by == "test-user"


def test_cache_yaml_load(tmp_path):
    from judgearena.config import load_config

    yaml_path = tmp_path / "cache.yaml"
    yaml_path.write_text(
        "task: alpaca-eval\n"
        "model: {name: Dummy/a, baseline: Dummy/b}\n"
        "judge: {model: Dummy/j}\n"
        "cache:\n"
        "  store_root: /data/cache\n"
        "  cache_mode: refresh\n"
        "  cache_fetch: true\n"
        "  cache_push: true\n"
        "  pushed_by: yaml-user\n"
    )
    cfg = load_config(yaml_path)
    assert cfg.cache.store_root == "/data/cache"
    assert cfg.cache.cache_mode == "refresh"
    assert cfg.cache.cache_fetch is True
    assert cfg.cache.cache_push is True
    assert cfg.cache.pushed_by == "yaml-user"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"cache_fetch": True},
            "cache.store_root is required",
        ),
        (
            {"store_root": "/tmp", "cache_fetch": True, "cache_hf_repo": "  "},
            "cache_hf_repo must be non-empty",
        ),
        (
            {"store_root": "/tmp", "cache_create_pr": True},
            "cache_push is required",
        ),
        (
            {"store_root": "/tmp", "cache_mode": "off", "cache_fetch": True},
            "cache_fetch and cache_push cannot be enabled",
        ),
        (
            {"cache_mode": "refresh"},
            "cache.store_root is required when cache_mode is refresh",
        ),
        (
            {"store_root": "   "},
            "cache.store_root must be non-empty",
        ),
    ],
)
def test_cache_validation_rejects_invalid_combinations(kwargs, match):
    data = _base_generate()
    data["cache"] = kwargs
    with pytest.raises(ValidationError, match=match):
        RunConfig(**data)


def test_inference_cache_session_yields_none_without_store_root():
    from judgearena.config import inference_cache_session

    cfg = RunConfig(**_base_generate())
    with inference_cache_session(cfg) as cache:
        assert cache is None


def test_inference_cache_session_opens_cache(tmp_path, monkeypatch):
    from judgearena.config import inference_cache_session, inference_cache_task

    monkeypatch.setattr("getpass.getuser", lambda: "session-user")
    data = _base_generate()
    data["cache"] = {
        "store_root": str(tmp_path / "store"),
        "cache_mode": "refresh",
        "cache_fetch": True,
        "cache_push": True,
        "pushed_by": "session-user",
    }
    cfg = RunConfig(**data)
    with inference_cache_session(cfg) as cache:
        assert cache is not None
        assert cache.store_root == tmp_path / "store"
        assert cache.task == inference_cache_task(cfg)
        assert cache.mode == "refresh"
        assert cache.fetch is True
        assert cache.push is True
        assert cache.pushed_by == "session-user"
