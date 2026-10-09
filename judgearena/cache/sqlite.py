"""Content-addressed local SQLite caches."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, astuple, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Literal
from urllib.parse import quote, unquote

import pandas as pd

from judgearena.usage import RequestUsage, request_usage_from_json

COMPLETION_DB_NAME = "completions.db"
JUDGEMENT_DB_NAME = "judgements.db"
DESCRIPTOR_FILENAME = "metadata.json"

CacheKind = Literal["completions", "judgements"]


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cached_usage_to_json(usage: RequestUsage | None) -> str | None:
    """Persist reusable usage without the original request's dollar charge."""
    if usage is None:
        return None
    values = asdict(usage)
    values.pop("cost_usd")
    return stable_json_dumps(values)


def cached_usage_from_json(value: str | None) -> RequestUsage | None:
    """Restore tokens, discarding charges also present in older cache rows."""
    usage = request_usage_from_json(value)
    return replace(usage, cost_usd=None) if usage is not None else None


def descriptor_hash(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(descriptor).encode()).hexdigest()[:16]


def input_hash(input_text: str) -> str:
    return hashlib.sha256(input_text.encode()).hexdigest()


@dataclass(frozen=True, order=True)
class CacheFolder:
    """Cache folder location relative to the store root."""

    kind: CacheKind
    task: str
    provider: str
    model: str
    descriptor_hash: str

    @classmethod
    def parse(cls, relative_path: PurePath) -> CacheFolder | None:
        parts = relative_path.parts
        if len(parts) != len(fields(cls)):
            return None
        return cls(*map(unquote, parts))

    @property
    def path(self) -> PurePosixPath:
        return PurePosixPath(*(quote(part, safe="") for part in astuple(self)))

    @property
    def model_spec(self) -> str:
        return f"{self.provider}/{self.model}"


def cache_folder(
    store_root: Path | str,
    kind: CacheKind,
    task: str,
    model_spec: str,
    descriptor: dict[str, Any],
) -> Path:
    provider, model = model_spec.split("/", 1)
    folder = CacheFolder(kind, task, provider, model, descriptor_hash(descriptor))
    return Path(store_root) / folder.path


def write_descriptor(folder: Path, descriptor: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / DESCRIPTOR_FILENAME
    if path.exists():
        if json.loads(path.read_text()) != descriptor:
            raise ValueError(f"Descriptor does not match existing metadata at {path}.")
        return path

    path.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")
    return path


def read_descriptor(folder: Path) -> dict[str, Any]:
    descriptor = json.loads((folder / DESCRIPTOR_FILENAME).read_text())
    if folder.name != descriptor_hash(descriptor):
        raise ValueError(f"Descriptor hash does not match cache folder {folder}.")
    return descriptor


class _SQLiteCache:
    table: str
    schema: str
    output_column: str

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(self.schema)
                columns = {
                    row[1]
                    for row in self._conn.execute(f"PRAGMA table_info({self.table})")
                }
                if "usage_json" not in columns:
                    self._conn.execute(
                        f"ALTER TABLE {self.table} ADD COLUMN usage_json TEXT"
                    )
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

    def merge_from(self, other_db: Path) -> int:
        """Merge another cache database in place using pushed_at last-write-wins."""
        conn = self._connect()
        if not other_db.exists():
            return conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]

        columns = [row[1] for row in conn.execute(f"PRAGMA table_info({self.table})")]
        column_list = ", ".join(columns)
        updates = ", ".join(
            (
                f"{column} = CASE "
                f"WHEN excluded.{column} IS NULL "
                f"AND excluded.{self.output_column} = "
                f"{self.table}.{self.output_column} "
                f"THEN {self.table}.{column} ELSE excluded.{column} END"
                if column == "usage_json"
                else f"{column} = excluded.{column}"
            )
            for column in columns
            if column != "input_hash"
        )
        conn.execute("ATTACH DATABASE ? AS incoming", (str(other_db),))
        try:
            incoming_columns = {
                row[1]
                for row in conn.execute(f"PRAGMA incoming.table_info({self.table})")
            }
            projection = column_list
            if "usage_json" not in incoming_columns:
                projection = ", ".join(
                    "NULL AS usage_json" if column == "usage_json" else column
                    for column in columns
                )
            with conn:
                conn.execute(
                    f"""
                    INSERT INTO {self.table} ({column_list})
                    SELECT {projection} FROM incoming.{self.table} WHERE true
                    ON CONFLICT(input_hash) DO UPDATE SET {updates}
                    WHERE excluded.pushed_at > {self.table}.pushed_at
                    """
                )
        finally:
            conn.execute("DETACH DATABASE incoming")
        return conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()[0]

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
    output_column = "completion"
    schema = """
        CREATE TABLE IF NOT EXISTS completions (
            input_hash     TEXT PRIMARY KEY,
            input_text     TEXT NOT NULL,
            completion     TEXT NOT NULL,
            benchmark      TEXT NOT NULL,
            instruction_id TEXT NOT NULL,
            model           TEXT NOT NULL,
            pushed_by       TEXT NOT NULL,
            pushed_at       TEXT NOT NULL,
            run_id          TEXT NOT NULL,
            usage_json      TEXT
        )
    """

    def save(
        self,
        rows: pd.DataFrame,
        *,
        pushed_by: str,
        run_id: str | None = None,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        resolved_run_id = run_id or str(uuid.uuid4())
        values = [
            (
                input_hash(str(row["input_text"])),
                str(row["input_text"]),
                str(row["completion"]),
                str(row["benchmark"]),
                str(row["instruction_id"]),
                str(row["model"]),
                pushed_by,
                now,
                resolved_run_id,
                cached_usage_to_json(row.get("usage_json")),
            )
            for _, row in rows.iterrows()
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO completions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
    output_column = "judge_completion"
    schema = """
        CREATE TABLE IF NOT EXISTS judgements (
            input_hash       TEXT PRIMARY KEY,
            judge_input      TEXT NOT NULL,
            judge_completion TEXT NOT NULL,
            benchmark        TEXT NOT NULL,
            instruction_id   TEXT NOT NULL,
            model_a          TEXT NOT NULL,
            model_b          TEXT,
            judge            TEXT NOT NULL,
            top_logprobs      TEXT,
            -- direct/reversed relative to the source model order, when applicable
            orientation      TEXT,
            pushed_by        TEXT NOT NULL,
            pushed_at        TEXT NOT NULL,
            run_id           TEXT NOT NULL,
            usage_json       TEXT
        )
    """

    def save(
        self,
        rows: pd.DataFrame,
        *,
        pushed_by: str,
        run_id: str | None = None,
    ) -> int:
        now = datetime.now(UTC).isoformat()
        resolved_run_id = run_id or str(uuid.uuid4())
        values = [
            (
                input_hash(str(row["judge_input"])),
                str(row["judge_input"]),
                str(row["judge_completion"]),
                str(row["benchmark"]),
                str(row["instruction_id"]),
                str(row["model_a"]),
                None if pd.isna(row.get("model_b")) else str(row["model_b"]),
                str(row["judge"]),
                (
                    stable_json_dumps(row["top_logprobs"])
                    if row.get("top_logprobs") is not None
                    else None
                ),
                row.get("orientation"),
                pushed_by,
                now,
                resolved_run_id,
                cached_usage_to_json(row.get("usage_json")),
            )
            for _, row in rows.iterrows()
        ]
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO judgements "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
