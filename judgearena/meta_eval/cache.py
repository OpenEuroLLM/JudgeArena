"""SQLite-backed cache for meta-evaluation judge annotations."""

from __future__ import annotations

import sqlite3
from dataclasses import astuple, dataclass, field, fields
from datetime import UTC, datetime
from itertools import groupby
from pathlib import Path

from judgearena.utils import data_root

DEFAULT_DB_DIR = data_root / "cache" / "db"


@dataclass(frozen=True)
class AnnotationEntry:
    benchmark: str
    instruction_id: str
    model_a: str
    model_b: str
    judge: str
    judge_input: str
    judge_completion: str
    reasoning_content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    date: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(field_info.name for field_info in fields(cls))

    @classmethod
    def key_fields(cls) -> tuple[str, ...]:
        return ("benchmark", "instruction_id", "model_a", "model_b", "judge")


@dataclass(frozen=True)
class AnnotationKey:
    benchmark: str
    instruction_id: str
    model_a: str
    model_b: str
    judge: str

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(field_info.name for field_info in fields(cls))


def _db_path(db_dir: Path, benchmark: str, judge: str) -> Path:
    return db_dir / benchmark / f"{judge.replace('/', '_')}.db"


class AnnotationCache:
    """Persistent per-battle cache matching the original meta-eval pipeline."""

    def __init__(self, db_dir: Path | str = DEFAULT_DB_DIR) -> None:
        self._db_dir = Path(db_dir)
        self._connections: dict[tuple[str, str], sqlite3.Connection] = {}

    def batch_get_annotations(
        self, keys: list[AnnotationKey]
    ) -> list[AnnotationEntry | None]:
        column_names = ", ".join(AnnotationEntry.field_names())
        results = []
        for key in keys:
            row = (
                self._connection(key.benchmark, key.judge)
                .execute(
                    f"SELECT {column_names} FROM annotations "
                    f"WHERE {self._where_clause()}",
                    astuple(key),
                )
                .fetchone()
            )
            results.append(AnnotationEntry(*row) if row else None)
        return results

    def batch_put(self, entries: list[AnnotationEntry]) -> None:
        if not entries:
            return
        column_names = ", ".join(AnnotationEntry.field_names())
        placeholders = ", ".join("?" for _ in AnnotationEntry.field_names())
        sql = (
            f"INSERT OR REPLACE INTO annotations ({column_names}) "
            f"VALUES ({placeholders})"
        )

        def cache_partition(entry: AnnotationEntry) -> tuple[str, str]:
            return entry.benchmark, entry.judge

        for (benchmark, judge), group in groupby(
            sorted(entries, key=cache_partition),
            key=cache_partition,
        ):
            connection = self._connection(benchmark, judge)
            connection.executemany(sql, [astuple(entry) for entry in group])
            connection.commit()

    def close(self) -> None:
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()

    def _connection(self, benchmark: str, judge: str) -> sqlite3.Connection:
        key = (benchmark, judge)
        if key not in self._connections:
            path = _db_path(self._db_dir, benchmark, judge)
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(
                str(path),
                check_same_thread=False,
                timeout=30,
            )
            connection.execute("PRAGMA journal_mode=WAL")
            self._connections[key] = connection
            self._create_table(connection)
        return self._connections[key]

    @staticmethod
    def _create_table(connection: sqlite3.Connection) -> None:
        integer_fields = {"input_tokens", "output_tokens", "reasoning_tokens"}
        default_text_fields = {"reasoning_content", "date"}
        column_definitions = ", ".join(
            (
                f"{name} INTEGER NOT NULL DEFAULT 0"
                if name in integer_fields
                else (
                    f"{name} TEXT NOT NULL DEFAULT ''"
                    if name in default_text_fields
                    else f"{name} TEXT NOT NULL"
                )
            )
            for name in AnnotationEntry.field_names()
        )
        key_columns = ", ".join(AnnotationEntry.key_fields())
        connection.execute(
            "CREATE TABLE IF NOT EXISTS annotations "
            f"({column_definitions}, UNIQUE ({key_columns}))"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_annotation_key "
            f"ON annotations ({key_columns})"
        )
        migrations = [
            "ALTER TABLE annotations ADD COLUMN "
            "reasoning_content TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE annotations ADD COLUMN input_tokens INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE annotations ADD COLUMN output_tokens INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE annotations ADD COLUMN "
            "reasoning_tokens INTEGER NOT NULL DEFAULT 0",
        ]
        for migration in migrations:
            try:
                connection.execute(migration)
            except sqlite3.OperationalError:
                pass
        connection.commit()

    @staticmethod
    def _where_clause() -> str:
        return " AND ".join(f"{column} = ?" for column in AnnotationKey.field_names())
