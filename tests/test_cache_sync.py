import json

import pandas as pd
import pytest

from judgearena.cache_sync import main
from judgearena.store_sqlite import (
    INFERENCE_DB_NAME,
    SQLiteInferenceStore,
    descriptor_hash,
    store_folder,
    write_store_metadata,
)

REPO_ID = "org/cache"
CELL_CONFIG = {"task": "arena", "model_spec": "VLLM/Qwen/judge"}
CELL_HASH = descriptor_hash(CELL_CONFIG)
CELL_PREFIX = f"inference/arena/VLLM/Qwen%2Fjudge/{CELL_HASH}"
DB_IN_REPO = f"{CELL_PREFIX}/{INFERENCE_DB_NAME}"
METADATA_IN_REPO = f"{CELL_PREFIX}/metadata.json"


def _write_cell(store_root):
    folder = store_folder(store_root, "arena", "VLLM/Qwen/judge", CELL_HASH)
    write_store_metadata(folder, CELL_CONFIG)
    with SQLiteInferenceStore(folder / INFERENCE_DB_NAME) as store:
        store.save_outputs(
            pd.DataFrame(
                {
                    "input_hash": ["h1"],
                    "input_text": ["input"],
                    "output_text": ["output"],
                }
            ),
            pushed_by="test",
        )
    return folder / INFERENCE_DB_NAME


def test_fetch_requires_filter(tmp_path):
    with pytest.raises(SystemExit, match="1"):
        main(["fetch", "--store_root", str(tmp_path)])


def test_fetch_bootstraps_filtered_remote_cell(fake_hub, tmp_path):
    remote_db = _write_cell(tmp_path / "remote")
    fake_hub.files[DB_IN_REPO] = remote_db.read_bytes()
    fake_hub.files[METADATA_IN_REPO] = json.dumps(CELL_CONFIG).encode()
    fake_hub.head = "initial"
    local_root = tmp_path / "local"

    main(
        [
            "fetch",
            "--store_root",
            str(local_root),
            "--cache_hf_repo",
            REPO_ID,
            "--task",
            "arena",
        ]
    )

    with SQLiteInferenceStore(
        store_folder(local_root, "arena", "VLLM/Qwen/judge", CELL_HASH)
        / INFERENCE_DB_NAME
    ) as store:
        assert store.query()["output_text"].tolist() == ["output"]


def test_push_uploads_local_cells(fake_hub, tmp_path):
    fake_hub.head = "initial"
    _write_cell(tmp_path)

    main(
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

    assert DB_IN_REPO in fake_hub.files
    assert METADATA_IN_REPO in fake_hub.files
