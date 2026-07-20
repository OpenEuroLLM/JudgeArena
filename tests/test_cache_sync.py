import getpass
import json
from pathlib import Path

import pandas as pd
import pytest

from judgearena import cache_sync
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


def _local_cell_db(tmp_path) -> Path:
    cell_dir = store_folder(tmp_path, "arena", MODEL_SPEC, CELL_CONFIG_HASH)
    return cell_dir / INFERENCE_DB_NAME


def _write_inference(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with SQLiteInferenceStore(path) as store:
        store.save_outputs(
            pd.DataFrame(
                {
                    "input_hash": ["local"],
                    "input_text": ["input-local"],
                    "output_text": ["L"],
                }
            ),
            pushed_by="test",
        )


def test_fetch_requires_filter(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cache_sync.main(["fetch", "--store_root", str(tmp_path)])
    assert exc.value.code == 1
    assert "requires a path filter" in capsys.readouterr().err


def test_fetch_rejects_invalid_filter_gaps(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cache_sync.main(
            [
                "fetch",
                "--store_root",
                str(tmp_path),
                "--provider",
                "VLLM",
            ]
        )
    assert exc.value.code == 1
    assert "requires --task" in capsys.readouterr().err


def test_fetch_bootstraps_filtered_remote_cells(fake_hub, tmp_path):
    remote = tmp_path / "remote.db"
    _write_inference(remote)
    fake_hub.files[PATH_IN_REPO] = remote.read_bytes()
    fake_hub.files[METADATA_IN_REPO] = json.dumps(CELL_CONFIG).encode("utf-8")
    fake_hub.head = "initial"

    store_root = tmp_path / "store"
    cache_sync.main(
        [
            "fetch",
            "--store_root",
            str(store_root),
            "--cache_hf_repo",
            REPO_ID,
            "--task",
            "arena",
        ]
    )
    local_db = _local_cell_db(store_root)
    assert local_db.exists()
    assert (local_db.parent / "metadata.json").exists()


def test_push_uploads_local_cells(fake_hub, tmp_path):
    local_db = _local_cell_db(tmp_path)
    _write_inference(local_db)
    write_store_metadata(local_db.parent, CELL_CONFIG)
    fake_hub.head = "initial"

    cache_sync.main(
        [
            "push",
            "--store_root",
            str(tmp_path),
            "--cache_hf_repo",
            REPO_ID,
            "--task",
            "arena",
        ]
    )
    assert PATH_IN_REPO in fake_hub.files
    assert METADATA_IN_REPO in fake_hub.files
    assert fake_hub.commit_calls == 1


def test_push_defaults_pushed_by_to_current_user(fake_hub, tmp_path, monkeypatch):
    local_db = _local_cell_db(tmp_path)
    _write_inference(local_db)
    write_store_metadata(local_db.parent, CELL_CONFIG)
    fake_hub.head = "initial"
    observed: list[str] = []

    def capture_push(*args, **kwargs):
        observed.append(kwargs.get("pushed_by", args[3] if len(args) > 3 else None))

    monkeypatch.setattr(cache_sync, "push_cells", capture_push)
    monkeypatch.setattr(getpass, "getuser", lambda: "unit-test-user")

    cache_sync.main(
        [
            "push",
            "--store_root",
            str(tmp_path),
            "--cache_hf_repo",
            REPO_ID,
            "--task",
            "arena",
        ]
    )
    assert observed == ["unit-test-user"]


def test_push_create_pr(fake_hub, tmp_path):
    local_db = _local_cell_db(tmp_path)
    _write_inference(local_db)
    write_store_metadata(local_db.parent, CELL_CONFIG)
    fake_hub.head = "initial"

    cache_sync.main(
        [
            "push",
            "--store_root",
            str(tmp_path),
            "--cache_hf_repo",
            REPO_ID,
            "--task",
            "arena",
            "--create_pr",
        ]
    )
    assert PATH_IN_REPO not in fake_hub.files
