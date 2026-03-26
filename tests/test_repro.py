import json

import judgearena.repro as repro


def test_write_run_metadata_writes_expected_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(
        repro, "_get_dependency_versions", lambda *args, **kwargs: {"pytest": "test"}
    )
    monkeypatch.setattr(repro, "_get_git_hash", lambda *args, **kwargs: "a" * 40)

    (tmp_path / "annotations.csv").write_text("instruction_index,judge_completion\n0,a\n")
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
