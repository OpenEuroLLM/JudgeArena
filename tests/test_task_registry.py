"""Tests for declarative task loading, discovery, and static commands."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from judgearena import cli as cli_module
from judgearena.tasks.cli import run_task_command
from judgearena.tasks.registry import TaskDefinitionError, load_tasks


def _task_definition(task: str = "test-task") -> dict[str, object]:
    return {
        "schema_version": 1,
        "task": task,
        "task_version": 1,
        "description": "Test pairwise task.",
        "tags": ["pairwise", "test"],
        "dataset": {
            "adapter": "judgearena_tables",
            "sources": {
                "examples": {
                    "type": "huggingface_dataset",
                    "repo_id": "example/tasks",
                    "revision": "a" * 40,
                    "allow_patterns": [f"*{task}*"],
                }
            },
            "fields": {"id": "id", "instruction": "prompt"},
        },
        "protocol": {
            "runner": "pairwise",
            "generation": {"mode": "single_turn_chat"},
            "baseline": {
                "strategy": "task_default",
                "reference_id": "reference-output",
                "allow_runtime_override": True,
            },
            "judge": {
                "default_prompt": "default",
                "parser": "pairwise_preference",
                "default_swap_mode": "fixed",
                "allowed_swap_modes": ["fixed", "both"],
            },
            "scoring": {
                "adapter": "pairwise_win_rate",
                "primary_metric": "winrate",
                "higher_is_better": True,
            },
        },
    }


def _write_family(
    root: Path,
    *,
    family: str,
    filename: str,
    definition: dict[str, object] | str,
) -> Path:
    family_dir = root / family
    family_dir.mkdir(parents=True, exist_ok=True)
    path = family_dir / filename
    text = definition if isinstance(definition, str) else yaml.safe_dump(definition)
    path.write_text(text)
    return path


def test_packaged_registry_discovers_alpaca_eval():
    tasks = load_tasks()
    resolved = tasks["alpaca-eval"]

    assert list(tasks) == ["alpaca-eval"]
    assert resolved.spec.dataset.sources["examples"].revision == (
        "004c4a992956eeefffd36b63ade470f32fd0a582"
    )
    assert resolved.spec.protocol.baseline.reference_id == "gpt4_1106_preview"
    assert resolved.spec.protocol.scoring.primary_metric == "winrate"


def test_find_returns_none_for_unregistered_task():
    assert load_tasks().get("not-packaged-yet") is None


def test_registry_rejects_unpinned_remote_source(tmp_path):
    definition = _task_definition()
    definition["dataset"]["sources"]["examples"]["revision"] = "main"
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=definition,
    )

    with pytest.raises(TaskDefinitionError, match="revision"):
        load_tasks(tmp_path)


def test_registry_rejects_duplicate_yaml_keys(tmp_path):
    text = yaml.safe_dump(_task_definition()) + "task: duplicate\n"
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=text,
    )

    with pytest.raises(TaskDefinitionError, match="duplicate key 'task'"):
        load_tasks(tmp_path)


def test_registry_resolves_private_base_and_records_provenance(tmp_path):
    definition = _task_definition()
    child_task = definition.pop("task")
    definition.pop("description")
    definition["tags"] = ["base"]
    _write_family(
        tmp_path,
        family="example",
        filename="_base.yaml",
        definition=definition,
    )
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition={
            "extends": "_base.yaml",
            "task": child_task,
            "description": "Resolved child.",
            "tags": ["child"],
        },
    )

    resolved = load_tasks(tmp_path)["test-task"]

    assert resolved.spec.description == "Resolved child."
    assert resolved.spec.tags == ("child",)
    assert [item.path for item in resolved.provenance.resources] == [
        "example/_base.yaml",
        "example/test-task.yaml",
    ]
    assert len(resolved.provenance.resolved_sha256) == 64


def test_registry_rejects_inheritance_cycle(tmp_path):
    _write_family(
        tmp_path,
        family="example",
        filename="_a.yaml",
        definition={"extends": "_b.yaml"},
    )
    _write_family(
        tmp_path,
        family="example",
        filename="_b.yaml",
        definition={"extends": "_a.yaml"},
    )
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition={"extends": "_a.yaml", "task": "test-task"},
    )

    with pytest.raises(TaskDefinitionError, match="inheritance cycle"):
        load_tasks(tmp_path)


def test_registry_rejects_extends_path_escape(tmp_path):
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition={"extends": "../../_base.yaml", "task": "test-task"},
    )

    with pytest.raises(TaskDefinitionError, match="path escapes"):
        load_tasks(tmp_path)


def test_registry_rejects_duplicate_task_ids(tmp_path):
    for family in ("one", "two"):
        _write_family(
            tmp_path,
            family=family,
            filename=f"{family}.yaml",
            definition=_task_definition("same-task"),
        )

    with pytest.raises(TaskDefinitionError, match="Duplicate task ID 'same-task'"):
        load_tasks(tmp_path)


def test_registry_rejects_unknown_adapter_id(tmp_path):
    definition = _task_definition()
    definition["dataset"]["adapter"] = "missing_loader"
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=definition,
    )

    with pytest.raises(TaskDefinitionError, match="unknown dataset adapter"):
        load_tasks(tmp_path)


def test_official_outputs_must_reference_declared_source(tmp_path):
    definition = _task_definition()
    definition["protocol"]["baseline"] = {
        "strategy": "official_outputs",
        "source": "missing_outputs",
    }
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=definition,
    )

    with pytest.raises(TaskDefinitionError, match="not declared in dataset.sources"):
        load_tasks(tmp_path)


def test_resolved_hash_ignores_yaml_formatting(tmp_path):
    definition = _task_definition()
    path = _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=definition,
    )
    first = load_tasks(tmp_path)["test-task"]

    path.write_text("# formatting-only change\n" + yaml.safe_dump(definition))
    second = load_tasks(tmp_path)["test-task"]

    assert first.provenance.source_sha256 != second.provenance.source_sha256
    assert first.provenance.resolved_sha256 == second.provenance.resolved_sha256


def test_unknown_task_lists_registered_tasks(tmp_path, capsys):
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=_task_definition(),
    )
    tasks = load_tasks(tmp_path)

    with pytest.raises(SystemExit):
        run_task_command(["show", "missing"], tasks=tasks)
    assert "test-task" in capsys.readouterr().err


def test_task_commands_list_show_and_validate(tmp_path, capsys, caplog):
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=_task_definition(),
    )
    tasks = load_tasks(tmp_path)

    run_task_command(["list"], tasks=tasks)
    assert capsys.readouterr().out.startswith("test-task\tv1\t")

    run_task_command(["show", "test-task", "--resolved"], tasks=tasks)
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["task"] == "test-task"
    assert shown["_provenance"]["resolved_sha256"]

    run_task_command(["validate"], tasks=tasks)
    assert "Validated 1 task(s)." in capsys.readouterr().out


def test_main_cli_intercepts_task_commands(monkeypatch, capsys):
    def unexpected_run_config(_argv):
        raise AssertionError("task commands must not construct RunConfig")

    monkeypatch.setattr(cli_module, "build_run_config", unexpected_run_config)

    cli_module.cli(["tasks", "list"])

    assert "alpaca-eval" in capsys.readouterr().out
