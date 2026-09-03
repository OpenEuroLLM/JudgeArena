"""Tests for declarative task loading, discovery, and static commands."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from judgearena import cli as cli_module
from judgearena.benchmarks.registry import benchmark_runner_names
from judgearena.datasets.registry import (
    battle_dataset_names,
    instruction_dataset_names,
)
from judgearena.tasks.cli import run_task_command
from judgearena.tasks.registry import (
    AdapterCatalog,
    TaskDefinitionError,
    load_tasks,
    resolve_task,
)


def test_adapter_catalog_uses_owning_registries():
    catalog = AdapterCatalog()

    assert catalog.runners == benchmark_runner_names()
    assert catalog.instruction_datasets == instruction_dataset_names()
    assert catalog.battle_datasets == battle_dataset_names()


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
            "judge": {
                "default_prompt_preset": "default",
                "default_swap_mode": "fixed",
            },
            "scoring": {"metrics": [{"metric": "pairwise_win_rate"}]},
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
    tasks = load_tasks()
    alpaca = tasks["alpaca-eval"]
    arena_v01 = tasks["arena-hard-v0.1"]
    arena_v20 = tasks["arena-hard-v2.0"]

    elo_comparia = tasks["elo-comparia"]
    elo_lmarena = tasks["elo-lmarena"]
    m_arena_v01 = tasks["m-arena-hard-v0.1"]
    m_arena_eu = resolve_task(tasks, "m-arena-hard-v2.0-EU")
    assert m_arena_eu is not None
    mt_bench = tasks["mt-bench"]

    assert {
        "alpaca-eval-ja",
        "arena-hard-v0.1-ja",
        "arena-hard-v2.0-ja",
    } <= set(tasks)
    assert alpaca.spec.task_version == 2
    assert arena_v01.spec.task_version == 2
    assert arena_v20.spec.task_version == 2
    assert alpaca.spec.dataset.sources["tables"].revision == (
        "004c4a992956eeefffd36b63ade470f32fd0a582"
    )
    assert alpaca.spec.protocol.baseline.reference_id == "gpt4_1106_preview"
    assert arena_v01.spec.protocol.baseline.reference_id == "gpt-4-0314"
    assert arena_v20.spec.protocol.baseline.references["hard_prompt"] == (
        "o3-mini-2025-01-31"
    )
    assert elo_comparia.spec.protocol.runner == "elo"
    assert elo_comparia.spec.protocol.arena == "ComparIA"
    assert [metric.metric for metric in elo_comparia.spec.protocol.scoring.metrics] == [
        "pairwise_win_rate",
        "bradley_terry",
    ]
    assert elo_comparia.spec.dataset.sources["comparia"].revision == (
        "7a40bce496c1f2aa3be4001da85a49cb4743042b"
    )
    assert elo_lmarena.spec.protocol.arena == "LMArena"
    assert len(elo_lmarena.spec.dataset.sources) == 3
    fluency = resolve_task(tasks, "fluency-french")
    assert fluency is not None
    assert fluency.spec.protocol.generation.mode == "base_completion"
    assert fluency.spec.protocol.baseline.strategy == "runtime_required"
    assert fluency.spec.protocol.judge.default_prompt_preset == "fluency"
    assert fluency.selection is not None
    assert fluency.selection.values == ("french",)
    assert [resource.path for resource in arena_v20.provenance.resources] == [
        "arena_hard/_base.yaml",
        "arena_hard/arena-hard-v2.0.yaml",
    ]
    assert m_arena_v01.spec.dataset.sources["examples"].revision == (
        "ab393a96cd0b134a1acfa96e080af31e5e73a393"
    )
    assert m_arena_v01.spec.protocol.baseline.reference_id == (
        "CohereLabs/aya-expanse-8b"
    )
    assert m_arena_eu.definition_task == "m-arena-hard-v2.0"
    assert m_arena_eu.selection is not None
    assert m_arena_eu.selection.name == "EU"
    assert m_arena_eu.selection.values == (
        "cs",
        "de",
        "el",
        "en",
        "es",
        "fr",
        "it",
        "nl",
        "pl",
        "pt",
        "ro",
        "uk",
    )
    assert mt_bench.spec.protocol.runner == "mt_bench"
    assert mt_bench.spec.protocol.generation.mode == "multi_turn_chat"
    assert mt_bench.spec.protocol.baseline.reference_id == "gpt-3.5-turbo"
    assert mt_bench.spec.protocol.generation.default_max_out_tokens == 1024
    assert mt_bench.spec.protocol.generation.default_seed == 0
    assert mt_bench.spec.protocol.judge.default_prompt_preset == "fastchat-pairwise"
    assert mt_bench.spec.protocol.judge.default_swap_mode == "both"
    assert mt_bench.spec.protocol.judge.default_max_out_tokens == 2048
    assert mt_bench.spec.protocol.judge.reference_categories == (
        "math",
        "reasoning",
        "coding",
        "arena-hard-200",
    )
    assert mt_bench.spec.dataset.sources["benchmark"].revision == (
        "a4b674ca573c24143824ac7f60d9173e7081e37d"
    )
    assert alpaca.spec.protocol.scoring.metrics[0].metric == (
        "alpaca_eval_length_controlled"
    )


def test_official_pairwise_tasks_declare_their_protocol_contracts():
    tasks = load_tasks()
    alpaca = resolve_task(tasks, "alpaca-eval").spec.protocol
    arena_v01 = resolve_task(tasks, "arena-hard-v0.1").spec.protocol
    arena_v20 = resolve_task(tasks, "arena-hard-v2.0").spec.protocol

    assert (
        alpaca.baseline.reference_id,
        alpaca.judge.default_prompt_preset,
        alpaca.judge.default_swap_mode,
        alpaca.judge.default_temperature,
        alpaca.judge.default_max_out_tokens,
        alpaca.judge.default_top_logprobs,
        [(metric.metric, metric.group_by) for metric in alpaca.scoring.metrics],
    ) == (
        "gpt4_1106_preview",
        "alpaca-eval",
        "random",
        1.0,
        1,
        5,
        [("alpaca_eval_length_controlled", ())],
    )
    assert dict(alpaca.scoring.metrics[0].parameters) == {
        "calibration_repo_id": "tatsu-lab/alpaca_eval",
        "calibration_filename": "df_gamed.csv",
        "calibration_revision": "2edc6fad8be6b14ea7230aabfd08188da6b8b814",
        "gamed_weight": 0.1,
    }
    alpaca_ja = resolve_task(tasks, "alpaca-eval-ja").spec.protocol
    assert (
        alpaca_ja.judge.default_prompt_preset,
        alpaca_ja.judge.default_swap_mode,
        [metric.metric for metric in alpaca_ja.scoring.metrics],
    ) == (
        "default",
        "fixed",
        ["pairwise_win_rate", "length_controlled_winrate"],
    )
    assert (
        arena_v01.baseline.reference_id,
        arena_v01.judge.default_prompt_preset,
        arena_v01.judge.default_swap_mode,
        arena_v01.judge.default_temperature,
        arena_v01.judge.default_max_out_tokens,
        [metric.metric for metric in arena_v01.scoring.metrics],
    ) == ("gpt-4-0314", "arena-hard", "both", 0.0, 4096, ["arena_hard_v01"])
    for canonical_name, ja_name in (
        ("arena-hard-v0.1", "arena-hard-v0.1-ja"),
        ("arena-hard-v2.0", "arena-hard-v2.0-ja"),
    ):
        canonical = resolve_task(tasks, canonical_name).spec
        ja = resolve_task(tasks, ja_name).spec
        assert ja.dataset == canonical.dataset
        assert ja.protocol.baseline == canonical.protocol.baseline
        assert ja.protocol.judge.default_prompt_preset == "default"
        assert ja.protocol.judge.default_swap_mode == "fixed"
        assert [metric.metric for metric in ja.protocol.scoring.metrics] == [
            "pairwise_win_rate"
        ]
    assert arena_v20.baseline.references == {
        "hard_prompt": "o3-mini-2025-01-31",
        "coding": "o3-mini-2025-01-31",
        "math": "o3-mini-2025-01-31",
        "creative_writing": "gemini-2.0-flash-001",
    }
    assert arena_v20.judge.category_prompts == {
        "creative_writing": "arena-hard-creative"
    }
    assert (
        arena_v20.judge.default_swap_mode,
        arena_v20.judge.default_temperature,
        arena_v20.judge.default_max_out_tokens,
        [(metric.metric, metric.group_by) for metric in arena_v20.scoring.metrics],
    ) == ("both", 0.0, 16000, [("arena_hard_v20", ("category",))])


def test_find_returns_none_for_unregistered_task():
    assert load_tasks().get("not-packaged-yet") is None


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
    _write_family(
        tmp_path,
        family="family",
        filename="family-v1.yaml",
        definition=definition,
    )

    with pytest.raises(TaskDefinitionError, match="unknown values"):
        load_tasks(tmp_path)


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

    with pytest.raises(TaskDefinitionError, match="collides with an existing task"):
        load_tasks(tmp_path)


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


def test_registry_rejects_dataset_adapter_from_another_protocol(tmp_path):
    definition = _task_definition()
    definition["dataset"]["adapter"] = "arena_battles"
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=definition,
    )

    with pytest.raises(TaskDefinitionError, match="unknown dataset adapter"):
        load_tasks(tmp_path)


def test_registry_rejects_unknown_metric_id(tmp_path):
    definition = _task_definition()
    definition["protocol"]["scoring"]["metrics"] = [{"metric": "missing_metric"}]
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=definition,
    )

    with pytest.raises(TaskDefinitionError, match="unknown metric"):
        load_tasks(tmp_path)


def test_registry_validates_metric_parameters_with_source_path(tmp_path):
    definition = _task_definition()
    definition["protocol"]["scoring"]["metrics"] = [
        {"metric": "pairwise_win_rate", "parameters": {"soft": False}}
    ]
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=definition,
    )

    with pytest.raises(
        TaskDefinitionError,
        match=r"example/test-task.yaml: invalid metric 'pairwise_win_rate'.*unexpected",
    ):
        load_tasks(tmp_path)


def test_metric_parameters_are_preserved_in_resolved_task(tmp_path):
    definition = _task_definition()
    definition["protocol"]["scoring"]["metrics"] = [
        {"metric": "bradley_terry", "parameters": {"n_bootstraps": 2}}
    ]
    _write_family(
        tmp_path,
        family="example",
        filename="test-task.yaml",
        definition=definition,
    )

    task = load_tasks(tmp_path)["test-task"]

    assert task.spec.protocol.scoring.metrics[0].parameters == {"n_bootstraps": 2}
    assert task.spec.model_dump(mode="json")["protocol"]["scoring"]["metrics"][0][
        "parameters"
    ] == {"n_bootstraps": 2}


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


def test_scoring_metrics_reject_duplicate_names():
    definition = _task_definition()
    definition["protocol"]["scoring"]["metrics"] = [
        {"metric": "pairwise_win_rate"},
        {"metric": "pairwise_win_rate", "group_by": ["category"]},
    ]

    with pytest.raises(ValueError, match="duplicate names"):
        from judgearena.tasks.schema import TaskSpec

        TaskSpec.model_validate(definition)


def test_mt_bench_has_no_pipeline_specific_metric_whitelist(tmp_path):
    definition = _task_definition("mt-test")
    definition["protocol"] = {
        "runner": "mt_bench",
        "generation": {"mode": "multi_turn_chat"},
        "baseline": {
            "strategy": "task_default",
            "reference_id": "reference-output",
        },
        "judge": {
            "default_prompt_preset": "default",
            "default_swap_mode": "fixed",
            "turns_mode": "both",
            "fastchat_prompt_preset": "default",
            "fastchat_temperature": 0.0,
        },
        "scoring": {"metrics": [{"metric": "length_controlled_winrate"}]},
    }
    _write_family(
        tmp_path,
        family="mt",
        filename="mt-test.yaml",
        definition=definition,
    )

    task = load_tasks(tmp_path)["mt-test"]

    assert task.spec.protocol.scoring.metrics[0].metric == "length_controlled_winrate"
