"""Cache-aware inference boundary backed by configuration-scoped SQLite cells."""

from __future__ import annotations

import getpass
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from judgearena.log import get_logger
from judgearena.store_sqlite import (
    INFERENCE_DB_NAME,
    SQLiteInferenceStore,
    descriptor_hash,
    stable_json_dumps,
    store_folder,
    write_store_metadata,
)
from judgearena.store_sync import DEFAULT_CACHE_REPO, fetch_cells, push_cells

logger = get_logger(__name__)

CacheMode = Literal["use", "off", "refresh"]
_VALID_CACHE_MODES = frozenset({"use", "off", "refresh"})


class InferenceCache:
    """Context manager for one run-scoped inference cache session."""

    def __init__(
        self,
        store_root: Path | str,
        task: str,
        *,
        mode: CacheMode = "use",
        fetch: bool = False,
        push: bool = False,
        create_pr: bool = False,
        cache_hf_repo: str = DEFAULT_CACHE_REPO,
        pushed_by: str | None = None,
        repo_type: str = "dataset",
        revision: str = "main",
    ) -> None:
        if mode not in _VALID_CACHE_MODES:
            raise ValueError(
                f"Invalid cache mode {mode!r}; expected one of {sorted(_VALID_CACHE_MODES)}"
            )
        self.store_root = Path(store_root).expanduser()
        self.task = task
        self.mode = mode
        self.fetch = fetch
        self.push = push
        self.create_pr = create_pr
        self.cache_hf_repo = cache_hf_repo
        self.pushed_by = pushed_by or getpass.getuser()
        self.repo_type = repo_type
        self.revision = revision
        self.run_id = str(uuid.uuid4())
        self._stores: dict[tuple[str, str], SQLiteInferenceStore] = {}
        self._cell_folders: dict[tuple[str, str], Path] = {}
        self._fetched_cells: set[tuple[str, str]] = set()
        self._dirty_cells: set[tuple[str, str]] = set()
        self._closed = False

    def _cell_key(self, model_spec: str, descriptor: dict[str, Any]) -> tuple[str, str]:
        return model_spec, descriptor_hash(descriptor)

    def _cell_folder(self, model_spec: str, descriptor: dict[str, Any]) -> Path:
        key = self._cell_key(model_spec, descriptor)
        if key not in self._cell_folders:
            config_hash = key[1]
            self._cell_folders[key] = store_folder(
                self.store_root,
                self.task,
                model_spec,
                config_hash,
            )
        return self._cell_folders[key]

    def _open_store(
        self, model_spec: str, descriptor: dict[str, Any]
    ) -> SQLiteInferenceStore:
        key = self._cell_key(model_spec, descriptor)
        if key not in self._stores:
            folder = self._cell_folder(model_spec, descriptor)
            write_store_metadata(folder, descriptor)
            db_path = folder / INFERENCE_DB_NAME
            self._stores[key] = SQLiteInferenceStore(db_path)
            if self.fetch and key not in self._fetched_cells:
                fetch_cells(
                    self.cache_hf_repo,
                    self.store_root,
                    [db_path],
                    repo_type=self.repo_type,
                    revision=self.revision,
                    strict=False,
                )
                self._fetched_cells.add(key)
        return self._stores[key]

    def get_or_run(
        self,
        *,
        model_spec: str,
        descriptor: dict[str, Any],
        canonical_inputs: Sequence[str],
        original_inputs: Sequence[Any],
        miss_runner: Callable[[list[Any]], list[str]],
        row_metadata: Sequence[dict[str, Any] | None] | None = None,
        producer_metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Return outputs in caller order, deduplicating identical canonical inputs."""
        if len(canonical_inputs) != len(original_inputs):
            raise ValueError(
                "canonical_inputs and original_inputs must have equal length"
            )
        if not canonical_inputs:
            return []

        if row_metadata is not None and len(row_metadata) != len(original_inputs):
            raise ValueError("row_metadata length must match original_inputs")

        if self.mode == "off":
            outputs = miss_runner(list(original_inputs))
            if len(outputs) != len(original_inputs):
                raise ValueError("miss_runner returned unexpected number of outputs")
            return outputs

        input_hashes = [
            descriptor_hash(canonical, length=None) for canonical in canonical_inputs
        ]
        unique_order: list[str] = []
        seen_hashes: set[str] = set()
        for input_hash in input_hashes:
            if input_hash not in seen_hashes:
                unique_order.append(input_hash)
                seen_hashes.add(input_hash)

        hash_to_canonical = {
            input_hash: canonical
            for input_hash, canonical in zip(
                input_hashes, canonical_inputs, strict=True
            )
        }
        hash_to_original: dict[str, Any] = {}
        for input_hash, original in zip(input_hashes, original_inputs, strict=True):
            hash_to_original.setdefault(input_hash, original)

        store = self._open_store(model_spec, descriptor)
        cell_key = self._cell_key(model_spec, descriptor)

        if self.mode == "refresh":
            missing_hashes = unique_order
            cached_by_hash: dict[str, str] = {}
        else:
            missing_hashes = store.missing(unique_order)
            cached_rows = store.query(
                [h for h in unique_order if h not in missing_hashes]
            )
            cached_by_hash = {
                row["input_hash"]: row["output_text"]
                for _, row in cached_rows.iterrows()
            }

        metadata_df = self._row_metadata_frame(
            input_hashes=input_hashes,
            row_metadata=row_metadata,
        )
        metadata_written = 0
        if missing_hashes:
            miss_inputs = [hash_to_original[h] for h in missing_hashes]
            new_outputs = miss_runner(miss_inputs)
            if len(new_outputs) != len(missing_hashes):
                raise ValueError("miss_runner returned unexpected number of outputs")

            producer_json = stable_json_dumps(producer_metadata or {})
            outputs_df = pd.DataFrame(
                {
                    "input_hash": missing_hashes,
                    "input_text": [hash_to_canonical[h] for h in missing_hashes],
                    "output_text": new_outputs,
                    "producer_metadata_json": [producer_json] * len(missing_hashes),
                }
            )
            if metadata_df is not None:
                _, metadata_written = store.save_outputs_and_metadata(
                    outputs_df,
                    metadata_df,
                    pushed_by=self.pushed_by,
                    run_id=self.run_id,
                    replace=self.mode == "refresh",
                )
            else:
                store.save_outputs(
                    outputs_df,
                    pushed_by=self.pushed_by,
                    run_id=self.run_id,
                    replace=self.mode == "refresh",
                )
            self._dirty_cells.add(cell_key)
            if self.mode == "refresh":
                cached_by_hash.update(
                    dict(zip(missing_hashes, new_outputs, strict=True))
                )
            else:
                cached_by_hash.update(store.outputs_by_hash(missing_hashes))

        elif metadata_df is not None:
            metadata_written = store.save_metadata(metadata_df, run_id=self.run_id)
        if metadata_written:
            self._dirty_cells.add(cell_key)
        return [cached_by_hash[h] for h in input_hashes]

    @staticmethod
    def _row_metadata_frame(
        *,
        input_hashes: list[str],
        row_metadata: Sequence[dict[str, Any] | None] | None,
    ) -> pd.DataFrame | None:
        if not row_metadata:
            return None
        rows = []
        for input_hash, metadata in zip(input_hashes, row_metadata, strict=True):
            if metadata is None:
                continue
            rows.append(
                {
                    "input_hash": input_hash,
                    "metadata_json": stable_json_dumps(metadata),
                }
            )
        if not rows:
            return None
        return pd.DataFrame(rows)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for store in self._stores.values():
            store.close()
        self._stores.clear()

    def __enter__(self) -> InferenceCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        dirty_cells = set(self._dirty_cells)
        cell_folders = dict(self._cell_folders)
        try:
            self.close()
        finally:
            if exc_type is None and self.push and dirty_cells:
                resolved_paths = [
                    cell_folders[key] / INFERENCE_DB_NAME
                    for key in dirty_cells
                    if key in cell_folders
                    and (cell_folders[key] / INFERENCE_DB_NAME).exists()
                ]
                if resolved_paths:
                    try:
                        push_cells(
                            self.cache_hf_repo,
                            self.store_root,
                            resolved_paths,
                            pushed_by=self.pushed_by,
                            repo_type=self.repo_type,
                            revision=self.revision,
                            create_pr=self.create_pr,
                            strict=False,
                        )
                    except Exception as push_exc:  # noqa: BLE001
                        logger.warning(
                            "Cache push failed after inference run %s: %s",
                            self.run_id,
                            push_exc,
                        )
