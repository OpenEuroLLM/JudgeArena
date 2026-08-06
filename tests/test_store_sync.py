import json
from pathlib import Path

import pytest

from judgearena import store_sync
from judgearena.store_sqlite import (
    INFERENCE_DB_NAME,
    SQLiteInferenceStore,
    descriptor_hash,
    store_folder,
    write_store_metadata,
)

REPO_ID = "org/cache"
CELL_CONFIG = {"task": "arena", "model_spec": "VLLM/Qwen/judge"}
CELL_CONFIG_HASH = descriptor_hash(CELL_CONFIG)
MODEL_SPEC = "VLLM/Qwen/judge"
PATH_IN_REPO = (
    f"inference/arena/VLLM/Qwen%2Fjudge/{CELL_CONFIG_HASH}/{INFERENCE_DB_NAME}"
)
METADATA_IN_REPO = f"inference/arena/VLLM/Qwen%2Fjudge/{CELL_CONFIG_HASH}/metadata.json"


def _local_cell_db(tmp_path, root: str = "store") -> Path:
    cell_dir = store_folder(
        Path(tmp_path) / root, "arena", MODEL_SPEC, CELL_CONFIG_HASH
    )
    return cell_dir / INFERENCE_DB_NAME


def _write_inference(path: Path, rows: list[dict]) -> None:
    with SQLiteInferenceStore(path) as store:
        conn = store._connect()
        conn.executemany(
            "INSERT OR REPLACE INTO inference "
            "(input_hash, input_text, output_text, producer_metadata_json, "
            "pushed_by, pushed_at, run_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    row["input_hash"],
                    row.get("input_text", f"input-{row['input_hash']}"),
                    row["output_text"],
                    row.get("producer_metadata_json", "{}"),
                    row.get("pushed_by", "test"),
                    row["pushed_at"],
                    row.get("run_id", "run"),
                )
                for row in rows
            ],
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _write_metadata(path: Path, rows: list[dict]) -> None:
    with SQLiteInferenceStore(path) as store:
        conn = store._connect()
        conn.executemany(
            "INSERT OR REPLACE INTO inference_metadata "
            "(input_hash, metadata_hash, metadata_json, observed_at, run_id) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    row["input_hash"],
                    row["metadata_hash"],
                    row["metadata_json"],
                    row["observed_at"],
                    row.get("run_id", "run"),
                )
                for row in rows
            ],
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _read_outputs(path: Path) -> dict[str, str]:
    with SQLiteInferenceStore(path) as store:
        frame = store.query()
    return dict(zip(frame["input_hash"], frame["output_text"], strict=True))


def _read_metadata(path: Path) -> set[tuple[str, str]]:
    with SQLiteInferenceStore(path) as store:
        frame = store.query_metadata()
    return {
        (row.input_hash, row.metadata_hash) for row in frame.itertuples(index=False)
    }


def test_validate_path_filters_enforces_hierarchy():
    assert store_sync.validate_path_filters(task="arena") == "inference/arena"
    assert (
        store_sync.validate_path_filters(task="arena", provider="VLLM")
        == "inference/arena/VLLM"
    )
    with pytest.raises(ValueError, match="requires --task"):
        store_sync.validate_path_filters(provider="VLLM")
    with pytest.raises(ValueError, match="requires --task and --provider"):
        store_sync.validate_path_filters(task="arena", model="Qwen/judge")
    with pytest.raises(ValueError, match="requires --task, --provider, and --model"):
        store_sync.validate_path_filters(task="arena", config_hash="abc123")
    with pytest.raises(ValueError, match="cache-cell directory"):
        store_sync.validate_path_filters(prefix="inference/arena/../outside")


def test_merge_dbs_unions_rows_and_newest_tuple_wins(tmp_path):
    older = tmp_path / "older.db"
    newer = tmp_path / "newer.db"
    _write_inference(
        older,
        [
            {
                "input_hash": "shared",
                "output_text": "old",
                "pushed_at": "2026-01-01",
                "run_id": "run-a",
            },
            {
                "input_hash": "only-old",
                "output_text": "old-only",
                "pushed_at": "2026-01-01",
            },
        ],
    )
    _write_inference(
        newer,
        [
            {
                "input_hash": "shared",
                "output_text": "new",
                "pushed_at": "2026-02-01",
                "run_id": "run-b",
            },
            {
                "input_hash": "only-new",
                "output_text": "new-only",
                "pushed_at": "2026-02-01",
            },
        ],
    )
    _write_metadata(
        older,
        [
            {
                "input_hash": "shared",
                "metadata_hash": "meta-a",
                "metadata_json": '{"a": 1}',
                "observed_at": "2026-01-01",
            }
        ],
    )
    _write_metadata(
        newer,
        [
            {
                "input_hash": "shared",
                "metadata_hash": "meta-b",
                "metadata_json": '{"b": 2}',
                "observed_at": "2026-02-01",
            }
        ],
    )

    merged = tmp_path / "merged.db"
    store_sync._merge_dbs([older, newer], merged)
    assert _read_outputs(merged) == {
        "shared": "new",
        "only-old": "old-only",
        "only-new": "new-only",
    }
    assert _read_metadata(merged) == {("shared", "meta-a"), ("shared", "meta-b")}


def test_merge_dbs_uses_output_hash_tiebreaker(tmp_path):
    left = tmp_path / "left.db"
    right = tmp_path / "right.db"
    _write_inference(
        left,
        [
            {
                "input_hash": "shared",
                "output_text": "zzz",
                "pushed_at": "2026-01-01",
                "run_id": "run-a",
            }
        ],
    )
    _write_inference(
        right,
        [
            {
                "input_hash": "shared",
                "output_text": "aaa",
                "pushed_at": "2026-01-01",
                "run_id": "run-a",
            }
        ],
    )
    merged = tmp_path / "merged.db"
    store_sync._merge_dbs([left, right], merged)
    assert _read_outputs(merged)["shared"] == "aaa"


def test_fetch_cell_merges_remote_into_local(fake_hub, tmp_path):
    remote = tmp_path / "remote.db"
    _write_inference(
        remote,
        [
            {
                "input_hash": "remote",
                "output_text": "R",
                "pushed_at": "2026-01-01",
            }
        ],
    )
    fake_hub.files[PATH_IN_REPO] = remote.read_bytes()
    fake_hub.files[METADATA_IN_REPO] = json.dumps(CELL_CONFIG).encode("utf-8")

    local = _local_cell_db(tmp_path)
    _write_inference(
        local,
        [
            {
                "input_hash": "local",
                "output_text": "L",
                "pushed_at": "2026-01-01",
            }
        ],
    )
    write_store_metadata(local.parent, CELL_CONFIG)
    assert store_sync.fetch_cell(REPO_ID, PATH_IN_REPO, local)
    assert _read_outputs(local) == {"local": "L", "remote": "R"}


def test_fetch_cell_rejects_metadata_mismatch_before_db_merge(fake_hub, tmp_path):
    remote = tmp_path / "remote.db"
    _write_inference(
        remote,
        [
            {
                "input_hash": "remote",
                "output_text": "R",
                "pushed_at": "2026-01-01",
            }
        ],
    )
    fake_hub.files[PATH_IN_REPO] = remote.read_bytes()
    fake_hub.files[METADATA_IN_REPO] = json.dumps(
        {**CELL_CONFIG, "task": "other"}
    ).encode("utf-8")

    local = _local_cell_db(tmp_path)
    _write_inference(
        local,
        [
            {
                "input_hash": "local",
                "output_text": "L",
                "pushed_at": "2026-01-01",
            }
        ],
    )
    write_store_metadata(local.parent, CELL_CONFIG)

    with pytest.raises(ValueError, match="does not match"):
        store_sync.fetch_cell(REPO_ID, PATH_IN_REPO, local)
    assert _read_outputs(local) == {"local": "L"}


def test_push_cell_retries_and_preserves_concurrent_rows(fake_hub, tmp_path):
    fake_hub.head = "initial"
    cell_dir = store_folder(tmp_path, "arena", MODEL_SPEC, CELL_CONFIG_HASH)
    local = cell_dir / INFERENCE_DB_NAME
    write_store_metadata(cell_dir, CELL_CONFIG)
    _write_inference(
        local,
        [
            {
                "input_hash": "local",
                "output_text": "L",
                "pushed_at": "2026-01-01",
            }
        ],
    )
    concurrent = tmp_path / "concurrent.db"
    _write_inference(
        concurrent,
        [
            {
                "input_hash": "remote",
                "output_text": "R",
                "pushed_at": "2026-02-01",
            }
        ],
    )

    def inject_concurrent_write():
        fake_hub.files[PATH_IN_REPO] = concurrent.read_bytes()
        fake_hub.head = "concurrent-head"

    fake_hub.pending_conflict = True
    fake_hub.inject = inject_concurrent_write
    store_sync.push_cell(
        REPO_ID,
        PATH_IN_REPO,
        local,
        pushed_by="alice",
    )

    assert fake_hub.commit_calls == 2
    uploaded = tmp_path / "uploaded.db"
    uploaded.write_bytes(fake_hub.files[PATH_IN_REPO])
    expected = {"local": "L", "remote": "R"}
    assert _read_outputs(uploaded) == expected
    assert _read_outputs(local) == expected
    assert METADATA_IN_REPO in fake_hub.files


def test_push_cell_requires_metadata(fake_hub, tmp_path):
    fake_hub.head = "initial"
    local = tmp_path / "local.db"
    _write_inference(
        local,
        [
            {
                "input_hash": "local",
                "output_text": "L",
                "pushed_at": "2026-01-01",
            }
        ],
    )
    with pytest.raises(FileNotFoundError, match="Missing metadata.json"):
        store_sync.push_cell(REPO_ID, PATH_IN_REPO, local, pushed_by="alice")
