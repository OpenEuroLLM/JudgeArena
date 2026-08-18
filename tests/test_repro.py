import json

import judgearena.artifacts.metadata as repro


def test_write_run_metadata_writes_expected_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(
        repro, "_get_dependency_versions", lambda *args, **kwargs: {"pytest": "test"}
    )
    monkeypatch.setattr(repro, "_get_git_hash", lambda *args, **kwargs: "a" * 40)

    (tmp_path / "annotations.csv").write_text(
        "instruction_index,judge_completion\n0,a\n"
    )
    (tmp_path / "results.json").write_text("{}")

    metadata_path = repro.write_run_metadata(
        output_dir=tmp_path,
        entrypoint="judgearena.test.entrypoint",
        run={"dataset": "alpaca-eval"},
        results={
            "num_battles": 3,
            "preferences": [0.0, 0.5, 1.0],
            "judge_score": float("nan"),
        },
        input_payloads={
            "instruction_index": [2, 1, 2],
            "instructions": ["i0", "i1", "i2"],
            "completions_A": ["a0", "a1", "a2"],
            "completions_B": ["b0", "b1", "b2"],
        },
        judge_system_prompt="system prompt",
        judge_user_prompt_template="user prompt",
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["schema_version"] == repro.METADATA_SCHEMA_VERSION
    assert metadata["entrypoint"] == "judgearena.test.entrypoint"
    assert metadata["results"]["num_battles"] == 3
    assert metadata["results"]["preferences_count"] == 3
    assert metadata["results"]["judge_score"] is None
    assert metadata["dataset_statistics"]["instruction_index_count"] == 3
    assert {artifact["path"] for artifact in metadata["artifacts"]} == {
        "annotations.csv",
        "results.json",
    }
    assert "extras" not in metadata
    assert metadata["git_hash"] == "a" * 40
    assert "instruction_indices_sha256" in metadata
    assert "judge_system_prompt_sha256" in metadata
    assert "judge_user_prompt_template_sha256" in metadata


def test_write_run_metadata_hashes_instruction_indices_as_set(tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "_get_dependency_versions", lambda *args, **kwargs: {})
    monkeypatch.setattr(repro, "_get_git_hash", lambda *args, **kwargs: None)

    metadata_path_a = repro.write_run_metadata(
        output_dir=tmp_path / "run_a",
        entrypoint="judgearena.test.entrypoint",
        run={"dataset": "alpaca-eval"},
        input_payloads={"instruction_index": [9, 1, 5, 9]},
    )
    metadata_path_b = repro.write_run_metadata(
        output_dir=tmp_path / "run_b",
        entrypoint="judgearena.test.entrypoint",
        run={"dataset": "alpaca-eval"},
        input_payloads={"instruction_index": [5, 9, 1]},
    )

    metadata_a = json.loads(metadata_path_a.read_text())
    metadata_b = json.loads(metadata_path_b.read_text())
    assert (
        metadata_a["instruction_indices_sha256"]
        == metadata_b["instruction_indices_sha256"]
    )


def test_write_run_metadata_hashes_named_judge_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "_get_dependency_versions", lambda *args, **kwargs: {})
    monkeypatch.setattr(repro, "_get_git_hash", lambda *args, **kwargs: None)

    metadata_path = repro.write_run_metadata(
        output_dir=tmp_path,
        entrypoint="judgearena.test.entrypoint",
        run={"dataset": "mt-bench"},
        judge_prompts={
            "single_turn": {
                "system_prompt": "single system",
                "user_prompt_template": "single user",
            },
            "multi_turn": {
                "system_prompt": "multi system",
                "user_prompt_template": "multi user",
            },
        },
    )

    metadata = json.loads(metadata_path.read_text())
    assert set(metadata["judge_prompts"]) == {"single_turn", "multi_turn"}
    assert metadata["judge_prompts"]["single_turn"] == {
        "system_prompt_sha256": repro._hash_string_sha256("single system"),
        "user_prompt_template_sha256": repro._hash_string_sha256("single user"),
    }
    assert metadata["judge_prompts"]["multi_turn"] == {
        "system_prompt_sha256": repro._hash_string_sha256("multi system"),
        "user_prompt_template_sha256": repro._hash_string_sha256("multi user"),
    }


def test_write_run_metadata_omits_optional_fields_when_inputs_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(repro, "_get_dependency_versions", lambda *args, **kwargs: {})
    monkeypatch.setattr(repro, "_get_git_hash", lambda *args, **kwargs: None)

    metadata_path = repro.write_run_metadata(
        output_dir=tmp_path,
        entrypoint="judgearena.test.entrypoint",
        run={"dataset": "alpaca-eval"},
    )

    metadata = json.loads(metadata_path.read_text())
    assert metadata["dataset_statistics"] == {}
    assert metadata["artifacts"] == []
    assert "git_hash" not in metadata
    assert "instruction_indices_sha256" not in metadata
    assert "judge_system_prompt_sha256" not in metadata
    assert "judge_user_prompt_template_sha256" not in metadata


def test_write_run_metadata_records_packaged_task_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "_get_dependency_versions", lambda *args, **kwargs: {})
    monkeypatch.setattr(repro, "_get_git_hash", lambda *args, **kwargs: None)

    metadata_path = repro.write_run_metadata(
        output_dir=tmp_path,
        entrypoint="judgearena.test.entrypoint",
        run={"task": "alpaca-eval"},
    )

    task_definition = json.loads(metadata_path.read_text())["task_definition"]
    assert task_definition["schema_version"] == 1
    assert task_definition["task_version"] == 1
    assert len(task_definition["resolved_sha256"]) == 64
    assert task_definition["resources"][0]["path"] == ("alpaca_eval/alpaca-eval.yaml")
