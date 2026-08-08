"""Content-addressed local SQLite caches."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import pandas as pd

COMPLETION_DB_NAME = "completions.db"
JUDGEMENT_DB_NAME = "judgements.db"
DESCRIPTOR_FILENAME = "metadata.json"

CacheKind = Literal["completions", "judgements"]


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def descriptor_hash(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(descriptor).encode()).hexdigest()[:16]


def input_hash(input_text: str) -> str:
    return hashlib.sha256(input_text.encode()).hexdigest()


def cache_folder(
    store_root: Path | str,
    kind: CacheKind,
    task: str,
    model_spec: str,
    descriptor: dict[str, Any],
) -> Path:
    provider, model = model_spec.split("/", 1)
    return (
        Path(store_root)
        / kind
        / quote(task, safe="")
        / quote(provider, safe="")
        / quote(model, safe="")
        / descriptor_hash(descriptor)
    )


def write_descriptor(folder: Path, descriptor: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / DESCRIPTOR_FILENAME
    if path.exists():
        if json.loads(path.read_text()) != descriptor:
            raise ValueError(f"Descriptor does not match existing metadata at {path}.")
        return path

    path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")
    return path


class _SQLiteCache:
    table: str
    schema: str

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute(self.schema)
            self._conn.commit()
        return self._conn

    def _query(
        self,
        input_hashes: list[str] | None,
        conditions: list[str],
        params: list[Any],
    ) -> pd.DataFrame:
        if input_hashes is not None:
            if not input_hashes:
                return pd.read_sql(
                    f"SELECT * FROM {self.table} WHERE 0", self._connect()
                )
            placeholders = ",".join("?" * len(input_hashes))
            conditions.append(f"input_hash IN ({placeholders})")
            params.extend(input_hashes)

        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        return pd.read_sql(
            f"SELECT * FROM {self.table}{where} ORDER BY instruction_id",
            self._connect(),
            params=params,
        )

    def _delete(self, conditions: list[str], params: list[Any]) -> int:
        if not conditions:
            raise ValueError("Delete requires at least one filter.")
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._connect() as conn:
            cursor = conn.execute(f"DELETE FROM {self.table}{where}", params)
        return cursor.rowcount

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> _SQLiteCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class CompletionCache(_SQLiteCache):
    """Completion rows keyed by the exact rendered model input."""

    table = "completions"
    schema = """
        CREATE TABLE IF NOT EXISTS completions (
            input_hash     TEXT PRIMARY KEY,
            input_text     TEXT NOT NULL,
            completion     TEXT NOT NULL,
            benchmark      TEXT NOT NULL,
            instruction_id TEXT NOT NULL,
            model           TEXT NOT NULL,
            pushed_at       TEXT NOT NULL
        )
    """

    def save(self, rows: pd.DataFrame) -> int:
        now = datetime.now(UTC).isoformat()
        values = [
            (
                input_hash(str(row["input_text"])),
                str(row["input_text"]),
                str(row["completion"]),
                str(row["benchmark"]),
                str(row["instruction_id"]),
                str(row["model"]),
                now,
            )
            for _, row in rows.iterrows()
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO completions VALUES (?, ?, ?, ?, ?, ?, ?)",
                values,
            )
        return len(values)

    def query(
        self,
        input_hashes: list[str] | None = None,
        *,
        instruction_id: str | None = None,
        model: str | None = None,
    ) -> pd.DataFrame:
        conditions: list[str] = []
        params: list[Any] = []
        if instruction_id is not None:
            conditions.append("instruction_id = ?")
            params.append(str(instruction_id))
        if model is not None:
            conditions.append("model = ?")
            params.append(model)
        return self._query(input_hashes, conditions, params)

    def delete(
        self,
        *,
        instruction_id: str | None = None,
        model: str | None = None,
    ) -> int:
        conditions: list[str] = []
        params: list[Any] = []
        if instruction_id is not None:
            conditions.append("instruction_id = ?")
            params.append(str(instruction_id))
        if model is not None:
            conditions.append("model = ?")
            params.append(model)
        return self._delete(conditions, params)


class JudgementCache(_SQLiteCache):
    """Raw judge completions keyed by the exact rendered judge input."""

    table = "judgements"
    schema = """
        CREATE TABLE IF NOT EXISTS judgements (
            input_hash       TEXT PRIMARY KEY,
            judge_input      TEXT NOT NULL,
            judge_completion TEXT NOT NULL,
            benchmark        TEXT NOT NULL,
            instruction_id   TEXT NOT NULL,
            model_a          TEXT NOT NULL,
            model_b          TEXT NOT NULL,
            judge            TEXT NOT NULL,
            -- direct/reversed relative to the source model order, when applicable
            orientation      TEXT,
            pushed_at        TEXT NOT NULL
        )
    """

    def save(self, rows: pd.DataFrame) -> int:
        now = datetime.now(UTC).isoformat()
        values = [
            (
                input_hash(str(row["judge_input"])),
                str(row["judge_input"]),
                str(row["judge_completion"]),
                str(row["benchmark"]),
                str(row["instruction_id"]),
                str(row["model_a"]),
                str(row["model_b"]),
                str(row["judge"]),
                row.get("orientation"),
                now,
            )
            for _, row in rows.iterrows()
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO judgements "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
        return len(values)

    def query(
        self,
        input_hashes: list[str] | None = None,
        *,
        instruction_id: str | None = None,
        model: str | None = None,
    ) -> pd.DataFrame:
        conditions: list[str] = []
        params: list[Any] = []
        if instruction_id is not None:
            conditions.append("instruction_id = ?")
            params.append(str(instruction_id))
        if model is not None:
            conditions.append("(model_a = ? OR model_b = ?)")
            params.extend((model, model))
        return self._query(input_hashes, conditions, params)

    def delete(
        self,
        *,
        instruction_id: str | None = None,
        model: str | None = None,
    ) -> int:
        conditions: list[str] = []
        params: list[Any] = []
        if instruction_id is not None:
            conditions.append("instruction_id = ?")
            params.append(str(instruction_id))
        if model is not None:
            conditions.append("(model_a = ? OR model_b = ?)")
            params.extend((model, model))
        return self._delete(conditions, params)
