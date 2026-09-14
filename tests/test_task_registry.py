"""Tests for declarative task loading, discovery, and static commands."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from judgearena import cli as cli_module
from judgearena.tasks.cli import run_task_command
from judgearena.tasks.registry import TaskDefinitionError, load_tasks, resolve_task


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
            },
            "judge": {"default_prompt_preset": "default", "default_swap_mode": "fixed"},
            "scoring": {"adapter": "pairwise_win_rate"},
        },
    }


def _write_task(path: Path, definition: dict[str, object] | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = definition if isinstance(definition, str) else yaml.safe_dump(definition)
    path.write_text(text)
    return path


def test_packaged_registry_discovers_versioned_tasks():
    tasks = load_tasks()
    assert {"arena-hard-v0.1", "arena-hard-v2.0"} <= tasks.keys()
    assert tasks["elo-comparia"].spec.protocol.runner == "elo"
    assert tasks["mt-bench"].spec.protocol.runner == "mt_bench"


def test_registry_resolves_task_family_suffixes(tmp_path):
    definition = _task_definition("family-v1")
    definition["variants"] = {
        "selector": "language",
        "values": ["de", "en", "uk"],
        "groups": {"EU": ["de", "en", "uk"]},
    }
    _write_task(tmp_path / "family/family-v1.yaml", definition)
    tasks = load_tasks(tmp_path)

    single = resolve_task(tasks, "family-v1-uk")
    group = resolve_task(tasks, "family-v1-EU")

    assert single is not None and group is not None
    assert single.task == "family-v1-uk"
    assert single.definition_task == "family-v1"
    assert single.selection is not None
    assert single.selection.selector == "language"
    assert single.selection.values == ("uk",)
    assert group.selection is not None
    assert group.selection.name == "EU"
    assert group.selection.values == ("de", "en", "uk")
    assert list(tasks) == ["family-v1"]
    assert resolve_task(tasks, "family-v1-fr") is None


def test_registry_rejects_variant_group_with_unknown_value(tmp_path):
    definition = _task_definition("family-v1")
    definition["variants"] = {
        "selector": "language",
        "values": ["de"],
        "groups": {"EU": ["de", "fr"]},
    }
    _write_task(tmp_path / "family/family-v1.yaml", definition)

    with pytest.raises(TaskDefinitionError, match="unknown values"):
        load_tasks(tmp_path)


def test_registry_rejects_variant_id_collision(tmp_path):
    family = _task_definition("family")
    family["variants"] = {"selector": "subset", "values": ["mini"]}
    _write_task(tmp_path / "family/family.yaml", family)
    _write_task(tmp_path / "other/family-mini.yaml", _task_definition("family-mini"))

    with pytest.raises(TaskDefinitionError, match="collides with an existing task"):
        load_tasks(tmp_path)


def test_registry_rejects_unpinned_remote_source(tmp_path):
    definition = _task_definition()
    definition["dataset"]["sources"]["examples"]["revision"] = "main"
    _write_task(tmp_path / "example/test-task.yaml", definition)

    with pytest.raises(TaskDefinitionError, match="revision"):
        load_tasks(tmp_path)


def test_registry_rejects_duplicate_yaml_keys(tmp_path):
    text = yaml.safe_dump(_task_definition()) + "task: duplicate\n"
    _write_task(tmp_path / "example/test-task.yaml", text)

    with pytest.raises(TaskDefinitionError, match="duplicate key 'task'"):
        load_tasks(tmp_path)


def test_registry_resolves_private_base_and_records_provenance(tmp_path):
    definition = _task_definition()
    child_task = definition.pop("task")
    definition.pop("description")
    definition["tags"] = ["base"]
    _write_task(tmp_path / "example/_base.yaml", definition)
    _write_task(
        tmp_path / "example/test-task.yaml",
        {
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
    _write_task(tmp_path / "example/_a.yaml", {"extends": "_b.yaml"})
    _write_task(tmp_path / "example/_b.yaml", {"extends": "_a.yaml"})
    _write_task(
        tmp_path / "example/test-task.yaml", {"extends": "_a.yaml", "task": "test-task"}
    )

    with pytest.raises(TaskDefinitionError, match="inheritance cycle"):
        load_tasks(tmp_path)


def test_registry_rejects_extends_path_escape(tmp_path):
    _write_task(
        tmp_path / "example/test-task.yaml",
        {"extends": "../../_base.yaml", "task": "test-task"},
    )

    with pytest.raises(TaskDefinitionError, match="path escapes"):
        load_tasks(tmp_path)


def test_registry_rejects_duplicate_task_ids(tmp_path):
    for family in ("one", "two"):
        _write_task(tmp_path / family / f"{family}.yaml", _task_definition("same-task"))

    with pytest.raises(TaskDefinitionError, match="Duplicate task ID 'same-task'"):
        load_tasks(tmp_path)


def test_registry_rejects_dataset_adapter_from_another_protocol(tmp_path):
    definition = _task_definition()
    definition["dataset"]["adapter"] = "arena_battles"
    _write_task(tmp_path / "example/test-task.yaml", definition)

    with pytest.raises(TaskDefinitionError, match="unknown dataset adapter"):
        load_tasks(tmp_path)


def test_registry_rejects_scorer_from_another_protocol(tmp_path):
    definition = _task_definition()
    definition["protocol"]["scoring"]["adapter"] = "bradley_terry"
    _write_task(tmp_path / "example/test-task.yaml", definition)

    with pytest.raises(TaskDefinitionError, match="unknown scorer"):
        load_tasks(tmp_path)


def test_official_outputs_must_reference_declared_source(tmp_path):
    definition = _task_definition()
    definition["protocol"]["baseline"] = {
        "strategy": "official_outputs",
        "source": "missing_outputs",
    }
    _write_task(tmp_path / "example/test-task.yaml", definition)

    with pytest.raises(TaskDefinitionError, match="not declared in dataset.sources"):
        load_tasks(tmp_path)


def test_category_baseline_uses_declared_category_field(tmp_path):
    definition = _task_definition()
    definition["dataset"]["fields"]["category"] = "category"
    definition["protocol"]["baseline"] = {
        "strategy": "category_defaults",
        "category_field": "other_category",
        "references": {"test": "reference"},
    }
    _write_task(tmp_path / "example/test-task.yaml", definition)

    with pytest.raises(TaskDefinitionError, match="dataset.fields.category"):
        load_tasks(tmp_path)


def test_resolved_hash_ignores_yaml_formatting(tmp_path):
    definition = _task_definition()
    path = _write_task(tmp_path / "example/test-task.yaml", definition)
    first = load_tasks(tmp_path)["test-task"]

    path.write_text("# formatting-only change\n" + yaml.safe_dump(definition))
    second = load_tasks(tmp_path)["test-task"]

    assert first.provenance.source_sha256 != second.provenance.source_sha256
    assert first.provenance.resolved_sha256 == second.provenance.resolved_sha256


def test_unknown_task_lists_registered_tasks(tmp_path, capsys):
    _write_task(tmp_path / "example/test-task.yaml", _task_definition())
    tasks = load_tasks(tmp_path)

    with pytest.raises(SystemExit):
        run_task_command(["show", "missing"], tasks=tasks)
    assert "test-task" in capsys.readouterr().err


def test_task_commands_list_show_and_validate(tmp_path, capsys):
    _write_task(tmp_path / "example/test-task.yaml", _task_definition())
    tasks = load_tasks(tmp_path)

    run_task_command(["list"], tasks=tasks)
    assert capsys.readouterr().out.startswith("test-task\tv1\t")

    run_task_command(["show", "test-task", "--resolved"], tasks=tasks)
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["task"] == "test-task"
    assert shown["_provenance"]["resolved_sha256"]

    run_task_command(["validate"], tasks=tasks)
    assert "Validated 1 task(s)." in capsys.readouterr().out


def test_task_show_reports_resolved_selection(tmp_path, capsys):
    definition = _task_definition("family")
    definition["variants"] = {"selector": "language", "values": ["uk"]}
    _write_task(tmp_path / "family/family.yaml", definition)

    run_task_command(["show", "family-uk", "--resolved"], tasks=load_tasks(tmp_path))

    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["task"] == "family"
    assert shown["_selection"] == {
        "selector": "language",
        "name": "uk",
        "values": ["uk"],
    }
    assert shown["_provenance"]["resolved_sha256"]


def test_main_cli_intercepts_task_commands(monkeypatch, capsys):
    def unexpected_run_config(_argv):
        raise AssertionError("task commands must not construct RunConfig")

    monkeypatch.setattr(cli_module, "build_run_config", unexpected_run_config)

    cli_module.cli(["tasks", "list"])

    assert "alpaca-eval" in capsys.readouterr().out
