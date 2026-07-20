"""SQLite-backed unified inference cache store."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd

from judgearena.log import get_logger

logger = get_logger(__name__)

IN_QUERY_CHUNK_SIZE = 500
INFERENCE_DB_NAME = "inference.db"

INFERENCE_COLUMNS = (
    "input_hash",
    "input_text",
    "output_text",
    "producer_metadata_json",
    "pushed_by",
    "pushed_at",
    "run_id",
)
METADATA_COLUMNS = (
    "input_hash",
    "metadata_hash",
    "metadata_json",
    "observed_at",
    "run_id",
)


def stable_json_dumps(value: Any) -> str:
    """Return deterministic JSON for hashing and descriptor comparison."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def descriptor_hash(value: Any, *, length: int | None = 16) -> str:
    """Return a stable SHA-256 digest for a JSON-serializable descriptor."""
    digest = hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def metadata_hash(metadata: Any) -> str:
    """Return the full content hash for optional row-level metadata."""
    return descriptor_hash(metadata, length=None)


def sanitize_path_component(value: str) -> str:
    """Return a single safe path segment without separators or traversal."""
    normalized = str(value).strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"Invalid path component: {value!r}")
    for segment in normalized.replace("\\", "/").split("/"):
        if segment in {"", ".", ".."}:
            raise ValueError(f"Invalid path component: {value!r}")
    return quote(normalized, safe="-_.~")


def store_folder(
    store_root: Path | str,
    task: str,
    model_spec: str,
    config_hash: str,
) -> Path:
    """Return the local folder for one configuration-scoped inference cell."""
    provider, model_path = model_spec.split("/", 1)
    return (
        Path(store_root).expanduser()
        / "inference"
        / sanitize_path_component(task)
        / sanitize_path_component(provider)
        / sanitize_path_component(model_path)
        / sanitize_path_component(config_hash)
    )


def _normalize_metadata_json(value: Any) -> tuple[str, Any]:
    """Return canonical JSON text and parsed value for one metadata payload."""
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return stable_json_dumps(parsed), parsed


def _resolve_metadata_hash(
    parsed_metadata: Any,
    provided_hash: Any,
) -> str:
    """Return a metadata hash, ignoring missing or NaN caller values."""
    if provided_hash is not None and not (
        isinstance(provided_hash, float) and pd.isna(provided_hash)
    ):
        provided = str(provided_hash).strip()
        if provided:
            return provided
    return metadata_hash(parsed_metadata)


def write_store_metadata(folder: Path | str, config: dict) -> Path:
    """Write or validate the descriptor that identifies a configuration cell."""
    folder = Path(folder)
    expected_hash = descriptor_hash(config)
    if folder.name != expected_hash:
        raise ValueError(
            f"Cell folder {folder.name!r} does not match descriptor hash "
            f"{expected_hash!r}."
        )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "metadata.json"
    serialized = stable_json_dumps(config)
    if path.exists():
        existing = stable_json_dumps(json.loads(path.read_text(encoding="utf-8")))
        if existing != serialized:
            raise ValueError(
                f"Existing metadata at {path} does not match the requested descriptor."
            )
        return path
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


class SQLiteInferenceStore:
    """SQLite-backed store for content-addressed inference outputs."""

    def __init__(self, db_path: Path | str, *, readonly: bool = False) -> None:
        self.db_path = Path(db_path)
        self.readonly = readonly
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.readonly:
                if not self.db_path.exists():
                    raise FileNotFoundError(
                        f"Inference store not found: {self.db_path}"
                    )
                uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
                self._conn = sqlite3.connect(uri, uri=True)
                return self._conn

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS inference (
                    input_hash              TEXT PRIMARY KEY,
                    input_text              TEXT NOT NULL,
                    output_text             TEXT NOT NULL,
                    producer_metadata_json  TEXT NOT NULL,
                    pushed_by               TEXT NOT NULL,
                    pushed_at               TEXT NOT NULL,
                    run_id                  TEXT NOT NULL
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_metadata (
                    input_hash      TEXT NOT NULL,
                    metadata_hash   TEXT NOT NULL,
                    metadata_json   TEXT NOT NULL,
                    observed_at     TEXT NOT NULL,
                    run_id          TEXT NOT NULL,
                    PRIMARY KEY (input_hash, metadata_hash)
                )
            """)
            self._conn.commit()
        return self._conn

    def _insert_outputs(
        self,
        df: pd.DataFrame,
        *,
        pushed_by: str,
        run_id: str,
        replace: bool = False,
    ) -> int:
        required = {"input_hash", "input_text", "output_text"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame missing columns: {missing_cols}")

        now = datetime.now(UTC).isoformat()
        rows = [
            (
                row["input_hash"],
                row["input_text"],
                row["output_text"],
                row.get("producer_metadata_json", "{}"),
                pushed_by,
                now,
                run_id,
            )
            for _, row in df.iterrows()
        ]
        verb = "REPLACE" if replace else "IGNORE"
        conn = self._connect()
        written = 0
        for row in rows:
            cursor = conn.execute(
                f"INSERT OR {verb} INTO inference "
                "(input_hash, input_text, output_text, producer_metadata_json, "
                "pushed_by, pushed_at, run_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                row,
            )
            written += cursor.rowcount
        return written

    def _insert_metadata(
        self,
        df: pd.DataFrame,
        *,
        run_id: str,
    ) -> int:
        required = {"input_hash", "metadata_json"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame missing columns: {missing_cols}")

        now = datetime.now(UTC).isoformat()
        rows: list[tuple[str, str, str, str, str]] = []
        for _, row in df.iterrows():
            metadata_json, parsed_metadata = _normalize_metadata_json(
                row["metadata_json"]
            )
            rows.append(
                (
                    row["input_hash"],
                    _resolve_metadata_hash(
                        parsed_metadata,
                        row["metadata_hash"] if "metadata_hash" in df.columns else None,
                    ),
                    metadata_json,
                    now,
                    run_id,
                )
            )
        conn = self._connect()
        written = 0
        for row in rows:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO inference_metadata "
                "(input_hash, metadata_hash, metadata_json, observed_at, run_id) "
                "VALUES (?, ?, ?, ?, ?)",
                row,
            )
            written += cursor.rowcount
        return written

    def save_outputs(
        self,
        df: pd.DataFrame,
        *,
        pushed_by: str,
        run_id: str | None = None,
        replace: bool = False,
    ) -> int:
        """Insert inference rows, optionally replacing existing keys."""
        with self._connect():
            written = self._insert_outputs(
                df,
                pushed_by=pushed_by,
                run_id=run_id or str(uuid.uuid4()),
                replace=replace,
            )
        logger.info("Wrote %d inference rows to %s", written, self.db_path)
        return written

    def save_metadata(
        self,
        df: pd.DataFrame,
        *,
        run_id: str | None = None,
    ) -> int:
        """Associate optional row metadata with cached inference rows."""
        with self._connect():
            written = self._insert_metadata(
                df,
                run_id=run_id or str(uuid.uuid4()),
            )
        logger.info("Wrote %d metadata associations to %s", written, self.db_path)
        return written

    def save_outputs_and_metadata(
        self,
        outputs: pd.DataFrame,
        metadata: pd.DataFrame,
        *,
        pushed_by: str,
        run_id: str | None = None,
        replace: bool = False,
    ) -> tuple[int, int]:
        """Atomically save inference outputs and their metadata associations."""
        resolved_run_id = run_id or str(uuid.uuid4())
        with self._connect():
            outputs_written = self._insert_outputs(
                outputs,
                pushed_by=pushed_by,
                run_id=resolved_run_id,
                replace=replace,
            )
            metadata_written = self._insert_metadata(
                metadata,
                run_id=resolved_run_id,
            )
        logger.info(
            "Wrote %d inference rows and %d metadata associations to %s",
            outputs_written,
            metadata_written,
            self.db_path,
        )
        return outputs_written, metadata_written

    def query(self, input_hashes: list[str] | None = None) -> pd.DataFrame:
        """Return inference rows, optionally restricted to input hashes."""
        conn = self._connect()
        if input_hashes is None:
            return pd.read_sql(
                "SELECT * FROM inference ORDER BY input_hash",
                conn,
            )
        if not input_hashes:
            return pd.read_sql("SELECT * FROM inference WHERE 0", conn)

        frames: list[pd.DataFrame] = []
        for chunk_start in range(0, len(input_hashes), IN_QUERY_CHUNK_SIZE):
            chunk = input_hashes[chunk_start : chunk_start + IN_QUERY_CHUNK_SIZE]
            placeholders = ",".join("?" * len(chunk))
            frames.append(
                pd.read_sql(
                    f"SELECT * FROM inference WHERE input_hash IN ({placeholders})"
                    " ORDER BY input_hash",
                    conn,
                    params=chunk,
                )
            )
        if len(frames) == 1:
            return frames[0]
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values("input_hash", kind="stable").reset_index(drop=True)

    def query_metadata(self, input_hashes: list[str] | None = None) -> pd.DataFrame:
        """Return metadata association rows."""
        conn = self._connect()
        if input_hashes is None:
            return pd.read_sql(
                "SELECT * FROM inference_metadata ORDER BY input_hash, metadata_hash",
                conn,
            )
        if not input_hashes:
            return pd.read_sql("SELECT * FROM inference_metadata WHERE 0", conn)

        frames: list[pd.DataFrame] = []
        for chunk_start in range(0, len(input_hashes), IN_QUERY_CHUNK_SIZE):
            chunk = input_hashes[chunk_start : chunk_start + IN_QUERY_CHUNK_SIZE]
            placeholders = ",".join("?" * len(chunk))
            frames.append(
                pd.read_sql(
                    f"SELECT * FROM inference_metadata "
                    f"WHERE input_hash IN ({placeholders}) "
                    "ORDER BY input_hash, metadata_hash",
                    conn,
                    params=chunk,
                )
            )
        if len(frames) == 1:
            return frames[0]
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values(
            ["input_hash", "metadata_hash"],
            kind="stable",
        ).reset_index(drop=True)

    def missing(self, input_hashes: list[str]) -> list[str]:
        """Return input hashes absent from the store, preserving caller order."""
        if not input_hashes:
            return []
        present = self.outputs_by_hash(input_hashes)
        return [value for value in input_hashes if value not in present]

    def outputs_by_hash(self, input_hashes: list[str]) -> dict[str, str]:
        """Return stored output text keyed by input hash."""
        if not input_hashes:
            return {}
        conn = self._connect()
        outputs: dict[str, str] = {}
        for chunk_start in range(0, len(input_hashes), IN_QUERY_CHUNK_SIZE):
            chunk = input_hashes[chunk_start : chunk_start + IN_QUERY_CHUNK_SIZE]
            placeholders = ",".join("?" * len(chunk))
            outputs.update(
                dict(
                    conn.execute(
                        f"SELECT input_hash, output_text FROM inference "
                        f"WHERE input_hash IN ({placeholders})",
                        chunk,
                    ).fetchall()
                )
            )
        return outputs

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SQLiteInferenceStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
