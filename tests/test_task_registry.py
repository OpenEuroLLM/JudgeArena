"""Tests for declarative task loading, discovery, and static commands."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from judgearena import cli as cli_module
from judgearena.tasks.cli import run_task_command
from judgearena.tasks.loader import TaskDefinitionError
from judgearena.tasks.registry import TaskRegistry, UnknownTaskError


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


def test_packaged_registry_discovers_versioned_tasks():
    registry = TaskRegistry()

    summary = registry.list()
    alpaca = registry.get("alpaca-eval")
    arena_v01 = registry.get("arena-hard-v0.1")
    arena_v20 = registry.get("arena-hard-v2.0")

    assert [task.task for task in summary] == [
        "alpaca-eval",
        "arena-hard-v0.1",
        "arena-hard-v2.0",
    ]
    assert alpaca.spec.dataset.sources["examples"].revision == (
        "004c4a992956eeefffd36b63ade470f32fd0a582"
    )
    assert alpaca.spec.protocol.baseline.reference_id == "gpt4_1106_preview"
    assert arena_v01.spec.protocol.baseline.reference_id == "gpt-4-0314"
    assert arena_v20.spec.protocol.baseline.references["hard_prompt"] == (
        "o3-mini-2025-01-31"
    )
    assert [resource.path for resource in arena_v20.provenance.resources] == [
        "arena_hard/_base.yaml",
        "arena_hard/arena-hard-v2.0.yaml",
    ]
    assert alpaca.spec.protocol.scoring.primary_metric == "winrate"


def test_find_returns_none_for_unregistered_task():
    assert TaskRegistry().find("not-packaged-yet") is None


def test_registry_resolves_task_family_suffixes(tmp_path):
    definition = _task_definition("family-v1")
    definition["variants"] = {
        "selector": "language",
        "values": ["de", "en", "uk"],
        "groups": {"EU": ["de", "en", "uk"]},
    }
    _write_family(
        tmp_path,
        family="family",
        filename="family-v1.yaml",
        definition=definition,
    )
    registry = TaskRegistry(tmp_path)

    single = registry.get("family-v1-uk")
    group = registry.get("family-v1-EU")

    assert single.task == "family-v1-uk"
    assert single.definition_task == "family-v1"
    assert single.selection is not None
    assert single.selection.selector == "language"
    assert single.selection.values == ("uk",)
    assert group.selection is not None
    assert group.selection.name == "EU"
    assert group.selection.values == ("de", "en", "uk")
    assert [summary.task for summary in registry.list()] == ["family-v1"]
    assert registry.find("family-v1-fr") is None


def test_registry_rejects_variant_group_with_unknown_value(tmp_path):
    definition = _task_definition("family-v1")
    definition["variants"] = {
        "selector": "language",
        "values": ["de"],
        "groups": {"EU": ["de", "fr"]},
    }
    _write_family(
        tmp_path,
        family="family",
        filename="family-v1.yaml",
        definition=definition,
    )

    with pytest.raises(TaskDefinitionError, match="unknown values"):
        TaskRegistry(tmp_path).validate_all()


def test_registry_rejects_variant_id_collision(tmp_path):
    family = _task_definition("family")
    family["variants"] = {
        "selector": "subset",
        "values": ["mini"],
    }
    _write_family(
        tmp_path,
        family="family",
        filename="family.yaml",
        definition=family,
    )
    _write_family(
        tmp_path,
        family="other",
        filename="family-mini.yaml",
        definition=_task_definition("family-mini"),
    )

    with pytest.raises(TaskDefinitionError, match="Variant task ID"):
        TaskRegistry(tmp_path).validate_all()


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
        TaskRegistry(tmp_path).validate_all()


def test_registry_rejects_duplicate_yaml_keys(tmp_path):
    text = yaml.safe_dump(_task_definition()) + "task: duplicate\n"
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=text,
    )

    with pytest.raises(TaskDefinitionError, match="duplicate key 'task'"):
        TaskRegistry(tmp_path).validate_all()


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

    resolved = TaskRegistry(tmp_path).get("test-task")

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
        TaskRegistry(tmp_path).validate_all()


def test_registry_rejects_extends_path_escape(tmp_path):
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition={"extends": "../../_base.yaml", "task": "test-task"},
    )

    with pytest.raises(TaskDefinitionError, match="path escapes"):
        TaskRegistry(tmp_path).validate_all()


def test_registry_rejects_duplicate_task_ids(tmp_path):
    for family in ("one", "two"):
        _write_family(
            tmp_path,
            family=family,
            filename=f"{family}.yaml",
            definition=_task_definition("same-task"),
        )

    with pytest.raises(TaskDefinitionError, match="Duplicate task ID 'same-task'"):
        TaskRegistry(tmp_path).validate_all()


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
        TaskRegistry(tmp_path).validate_all()


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
        TaskRegistry(tmp_path).validate_all()


def test_category_baseline_uses_declared_category_field(tmp_path):
    definition = _task_definition()
    definition["dataset"]["fields"]["category"] = "category"
    definition["protocol"]["baseline"] = {
        "strategy": "category_defaults",
        "category_field": "other_category",
        "references": {"test": "reference"},
    }
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=definition,
    )

    with pytest.raises(TaskDefinitionError, match="dataset.fields.category"):
        TaskRegistry(tmp_path).validate_all()


def test_resolved_hash_ignores_yaml_formatting(tmp_path):
    definition = _task_definition()
    path = _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=definition,
    )
    first = TaskRegistry(tmp_path).get("test-task")

    path.write_text("# formatting-only change\n" + yaml.safe_dump(definition))
    second = TaskRegistry(tmp_path).get("test-task")

    assert first.provenance.source_sha256 != second.provenance.source_sha256
    assert first.provenance.resolved_sha256 == second.provenance.resolved_sha256


def test_unknown_task_error_lists_registered_tasks(tmp_path):
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=_task_definition(),
    )

    with pytest.raises(UnknownTaskError, match="test-task"):
        TaskRegistry(tmp_path).get("missing")


def test_task_commands_list_show_and_validate(tmp_path, capsys):
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=_task_definition(),
    )
    registry = TaskRegistry(tmp_path)

    run_task_command(["list"], registry=registry)
    assert capsys.readouterr().out.startswith("test-task\tv1\t")

    run_task_command(["show", "test-task", "--resolved"], registry=registry)
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["task"] == "test-task"
    assert shown["_provenance"]["resolved_sha256"]

    run_task_command(["validate"], registry=registry)
    assert capsys.readouterr().out == "Validated 1 task(s).\n"


def test_task_show_reports_resolved_selection(tmp_path, capsys):
    definition = _task_definition("family")
    definition["variants"] = {
        "selector": "language",
        "values": ["uk"],
    }
    _write_family(
        tmp_path,
        family="family",
        filename="family.yaml",
        definition=definition,
    )

    run_task_command(
        ["show", "family-uk", "--resolved"], registry=TaskRegistry(tmp_path)
    )

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
