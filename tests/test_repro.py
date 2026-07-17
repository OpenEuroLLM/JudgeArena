import json

import pytest

import judgearena.repro as repro
from judgearena.config import RunConfig
from judgearena.prompts.registry import resolve_judge_prompt


def _resolved_prompt():
    return resolve_judge_prompt(preset="default")


def test_write_run_metadata_writes_compact_v2_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        repro, "_get_dependency_versions", lambda *args, **kwargs: {"pytest": "test"}
    )
    monkeypatch.setattr(repro, "_get_git_hash", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(repro, "_get_git_dirty", lambda *args, **kwargs: False)

    (tmp_path / "annotations.csv").write_text(
        "instruction_index,judge_completion\n0,a\n"
    )
    (tmp_path / "results.json").write_text("{}")
    (tmp_path / "stale.txt").write_text("from an earlier run")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("task: alpaca-eval\n")

    metadata_path = repro.write_run_metadata(
        output_dir=tmp_path,
        entrypoint="judgearena.test.entrypoint",
        context=repro.RunContext(
            identity=repro.RunIdentity(
                workflow="pairwise",
                task="alpaca-eval",
                model=repro.ModelIdentity(name="candidate", baseline="baseline"),
                judge=repro.JudgeIdentity(model="judge"),
            ),
            configuration=repro.ConfigurationMetadata.from_path(config_path),
        ),
        inputs=repro.InputMetadata.capture(
            dataset_revisions={"repo/dataset": "revision"},
            example_ids=[2, 1, 2],
            content={"instructions": ["two", "one", "two"]},
            judgment_count=6,
        ),
        judge_prompt=_resolved_prompt(),
        prompt_variants=[
            {
                "name": "single",
                "system_prompt": "system",
                "user_prompt_template": "judge {answer}",
            }
        ],
        artifacts={
            "annotations": tmp_path / "annotations.csv",
            "results": tmp_path / "results.json",
        },
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata_path.name == "run-metadata.v2.json"
    assert metadata["schema_version"] == repro.METADATA_SCHEMA_VERSION
    assert metadata["identity"] == {
        "workflow": "pairwise",
        "task": "alpaca-eval",
        "model": {"name": "candidate", "baseline": "baseline"},
        "judge": {"model": "judge"},
    }
    assert metadata["configuration"] == {
        "path": "config.yaml",
        "sha256": repro._hash_file_sha256(config_path),
    }
    assert metadata["execution"]["entrypoint"] == "judgearena.test.entrypoint"
    assert metadata["inputs"]["dataset_revisions"] == {"repo/dataset": "revision"}
    assert metadata["inputs"]["example_count"] == 3
    assert metadata["inputs"]["judgment_count"] == 6
    assert "example_ids_sha256" in metadata["inputs"]
    assert "content_sha256" in metadata["inputs"]
    assert "metrics" not in metadata
    assert metadata["code"] == {"git_commit": "a" * 40, "git_dirty": False}
    assert metadata["prompt"] == {
        "preset": "default",
        "source": "preset",
        "parser_mode": "score",
        "delegated": False,
        "system_path": "system-prompt.txt",
        "user_path": "prompt.txt",
        "variants": [
            {
                "name": "single",
                "system_sha256": repro._hash_string_sha256("system"),
                "user_sha256": repro._hash_string_sha256("judge {answer}"),
            }
        ],
    }
    assert metadata["artifacts"] == [
        {
            "kind": "annotations",
            "path": "annotations.csv",
            "size_bytes": (tmp_path / "annotations.csv").stat().st_size,
            "sha256": repro._hash_file_sha256(tmp_path / "annotations.csv"),
        },
        {
            "kind": "results",
            "path": "results.json",
            "size_bytes": (tmp_path / "results.json").stat().st_size,
            "sha256": repro._hash_file_sha256(tmp_path / "results.json"),
        },
    ]
    assert "stale.txt" not in {
        artifact["path"] for artifact in metadata["artifacts"]
    }
    assert "run" not in metadata
    assert "results" not in metadata
    assert "dataset_statistics" not in metadata
    assert "judge_system_prompt_sha256" not in metadata


def test_input_metadata_hashes_example_ids_as_normalized_set():
    inputs_a = repro.InputMetadata.capture(example_ids=[9, 1, 5])
    inputs_b = repro.InputMetadata.capture(example_ids=[5, 9, 1, 5])

    assert inputs_a.example_ids_sha256 == inputs_b.example_ids_sha256


def test_input_metadata_content_hash_changes_with_content():
    inputs_a = repro.InputMetadata.capture(content={"instructions": ["A"]})
    inputs_b = repro.InputMetadata.capture(content={"instructions": ["B"]})

    assert inputs_a.content_sha256 != inputs_b.content_sha256


def test_configuration_hash_changes_with_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("task: one\n")
    first = repro.ConfigurationMetadata.from_path(config_path)
    config_path.write_text("task: two\n")
    second = repro.ConfigurationMetadata.from_path(config_path)

    assert first.sha256 != second.sha256


def test_run_identity_uses_config_field_names():
    cfg = RunConfig(
        task="alpaca-eval",
        model={"name": "model-a", "baseline": "model-b"},
        judge={"model": "judge"},
    )

    identity = repro.RunIdentity.from_config(cfg, workflow="pairwise")

    assert identity.model_dump(exclude_none=True) == {
        "workflow": "pairwise",
        "task": "alpaca-eval",
        "model": {"name": "model-a", "baseline": "model-b"},
        "judge": {"model": "judge"},
    }


def test_prompt_hash_changes_with_resolved_prompt():
    default = repro.PromptMetadata.from_resolved(resolve_judge_prompt(preset="default"))
    fluency = repro.PromptMetadata.from_resolved(resolve_judge_prompt(preset="fluency"))
    with_explanation = repro.PromptMetadata.from_resolved(
        resolve_judge_prompt(preset="default_with_explanation")
    )

    assert default.system_sha256 != fluency.system_sha256
    assert default.user_sha256 != with_explanation.user_sha256


def test_write_run_metadata_omits_optional_fields_when_inputs_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(repro, "_get_dependency_versions", lambda *args, **kwargs: {})
    monkeypatch.setattr(repro, "_get_git_hash", lambda *args, **kwargs: None)
    monkeypatch.setattr(repro, "_get_git_dirty", lambda *args, **kwargs: None)

    metadata_path = repro.write_run_metadata(
        output_dir=tmp_path,
        entrypoint="judgearena.test.entrypoint",
        context=repro.RunContext(identity=repro.RunIdentity(workflow="pairwise")),
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["inputs"] == {}
    assert metadata["artifacts"] == []
    assert "metrics" not in metadata
    assert metadata["code"] == {}
    assert "prompt" not in metadata
    assert "configuration" not in metadata


def test_input_metadata_omits_redundant_judgment_count():
    inputs = repro.InputMetadata.capture(example_ids=[1, 2], judgment_count=2)

    assert inputs.example_count == 2
    assert inputs.judgment_count is None


def test_input_metadata_rejects_inconsistent_example_count():
    with pytest.raises(ValueError, match="must equal"):
        repro.InputMetadata.capture(
            example_ids=[1, 2],
            example_count=3,
        )
