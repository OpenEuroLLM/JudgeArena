import json

import pandas as pd
import pytest

from judgearena.store_sqlite import (
    IN_QUERY_CHUNK_SIZE,
    SQLiteInferenceStore,
    descriptor_hash,
    metadata_hash,
    sanitize_path_component,
    stable_json_dumps,
    store_folder,
    write_store_metadata,
)

CELL_CONFIG = {"task": "arena", "model_spec": "VLLM/Qwen/judge"}


def _outputs(hashes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "input_hash": hashes,
            "input_text": [f"input-{value}" for value in hashes],
            "output_text": [f"output-{value}" for value in hashes],
        }
    )


def test_inference_roundtrip_and_missing_order(tmp_path):
    hashes = ["h0", "h1", "h2"]
    with SQLiteInferenceStore(tmp_path / "inference.db") as store:
        store.save_outputs(_outputs(hashes), pushed_by="test")
        assert store.missing(hashes) == []
        assert store.missing([*hashes, "missing"]) == ["missing"]
        result = store.query(["h1", "absent", "h0"])
        assert store.query([]).empty

    assert result["input_hash"].tolist() == ["h0", "h1"]


def test_readonly_store_queries_without_schema_writes(tmp_path):
    db_path = tmp_path / "inference.db"
    with SQLiteInferenceStore(db_path) as store:
        store.save_outputs(_outputs(["h1"]), pushed_by="test")

    before = db_path.stat().st_mtime_ns
    with SQLiteInferenceStore(db_path, readonly=True) as store:
        assert store.outputs_by_hash(["h1", "missing"]) == {"h1": "output-h1"}
        assert store.missing(["h1", "missing"]) == ["missing"]
    assert db_path.stat().st_mtime_ns == before


def test_save_outputs_insert_only_vs_replace(tmp_path):
    db_path = tmp_path / "inference.db"
    with SQLiteInferenceStore(db_path) as store:
        assert store.save_outputs(_outputs(["same"]), pushed_by="alice") == 1
        assert (
            store.save_outputs(
                pd.DataFrame(
                    {
                        "input_hash": ["same"],
                        "input_text": ["input-same"],
                        "output_text": ["ignored"],
                    }
                ),
                pushed_by="bob",
            )
            == 0
        )
        assert store.query(["same"])["output_text"].iloc[0] == "output-same"
        assert (
            store.save_outputs(
                pd.DataFrame(
                    {
                        "input_hash": ["same"],
                        "input_text": ["input-same"],
                        "output_text": ["updated"],
                    }
                ),
                pushed_by="bob",
                replace=True,
            )
            == 1
        )
        result = store.query(["same"])

    assert result["output_text"].iloc[0] == "updated"
    assert result["pushed_by"].iloc[0] == "bob"


def test_many_to_one_metadata_is_idempotent(tmp_path):
    meta = {"question_id": "q-1", "role": "judge"}
    meta_json = stable_json_dumps(meta)
    with SQLiteInferenceStore(tmp_path / "inference.db") as store:
        store.save_outputs(_outputs(["h0"]), pushed_by="test")
        assert (
            store.save_metadata(
                pd.DataFrame({"input_hash": ["h0"], "metadata_json": [meta_json]}),
                run_id="run-a",
            )
            == 1
        )
        assert (
            store.save_metadata(
                pd.DataFrame({"input_hash": ["h0"], "metadata_json": [meta_json]}),
                run_id="run-b",
            )
            == 0
        )
        assert (
            store.save_metadata(
                pd.DataFrame(
                    {
                        "input_hash": ["h0"],
                        "metadata_hash": [metadata_hash(meta)],
                        "metadata_json": [meta_json],
                    }
                ),
                run_id="run-c",
            )
            == 0
        )
        rows = store.query_metadata(["h0"])

    assert len(rows) == 1
    assert json.loads(rows["metadata_json"].iloc[0]) == meta


def test_output_and_metadata_batch_rolls_back_atomically(tmp_path):
    with SQLiteInferenceStore(tmp_path / "inference.db") as store:
        with pytest.raises(ValueError):
            store.save_outputs_and_metadata(
                _outputs(["h0"]),
                pd.DataFrame(
                    {
                        "input_hash": ["h0"],
                        "metadata_json": ["not-json"],
                    }
                ),
                pushed_by="test",
            )

        assert store.query().empty
        assert store.query_metadata().empty


def test_chunked_query_preserves_input_independent_order(tmp_path, monkeypatch):
    hashes = [f"h{index}" for index in range(IN_QUERY_CHUNK_SIZE + 3)]
    with SQLiteInferenceStore(tmp_path / "inference.db") as store:
        store.save_outputs(_outputs(hashes), pushed_by="test")
        monkeypatch.setattr(
            "judgearena.store_sqlite.IN_QUERY_CHUNK_SIZE",
            2,
        )
        result = store.query(hashes)

    assert result["input_hash"].tolist() == sorted(hashes)


def test_write_store_metadata_rejects_misnamed_folder(tmp_path):
    config = {
        "descriptor_schema_version": "judgearena-cache/v1",
        "task": "arena-hard-v2.0",
        "model_spec": "VLLM/Qwen/Qwen3.5-9B",
    }
    wrong_folder = (
        tmp_path
        / "inference"
        / "arena"
        / "VLLM"
        / "Qwen%2FQwen3.5-9B"
        / "deadbeefdeadbeef"
    )
    with pytest.raises(ValueError, match="does not match descriptor hash"):
        write_store_metadata(wrong_folder, config)


def test_store_folder_and_metadata_validation(tmp_path):
    config = {
        "descriptor_schema_version": "judgearena-cache/v1",
        "task": "arena-hard-v2.0",
        "model_spec": "VLLM/Qwen/Qwen3.5-9B",
    }
    folder = store_folder(
        tmp_path,
        "arena-hard-v2.0",
        "VLLM/Qwen/Qwen3.5-9B",
        descriptor_hash(config),
    )
    metadata_path = write_store_metadata(folder, config)
    assert folder.parts[-3:-1] == ("VLLM", "Qwen%2FQwen3.5-9B")
    assert json.loads(metadata_path.read_text()) == config
    write_store_metadata(folder, config)
    with pytest.raises(ValueError, match="does not match"):
        write_store_metadata(folder, {**config, "task": "other"})


def test_sanitize_path_component_rejects_traversal():
    assert sanitize_path_component("Qwen/Qwen3.5-9B") == "Qwen%2FQwen3.5-9B"
    assert sanitize_path_component("Qwen--Qwen3.5-9B") != sanitize_path_component(
        "Qwen/Qwen3.5-9B"
    )
    with pytest.raises(ValueError, match="Invalid path component"):
        sanitize_path_component("..")
    with pytest.raises(ValueError, match="Invalid path component"):
        sanitize_path_component("foo/../bar")


def test_hash_helpers_are_stable():
    payload = {"b": 2, "a": 1}
    assert descriptor_hash(payload) == descriptor_hash({"a": 1, "b": 2})
    assert metadata_hash(payload) == descriptor_hash(payload, length=None)
