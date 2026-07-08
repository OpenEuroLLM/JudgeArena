"""SQLite-backed completion and judgement store.

One .db file per (task, model). Designed to live alongside a metadata.json:

  completions/{task}/{model_name}/{provider}/
      completions.db
      metadata.json

  judgements/{task}/{judge_name}/{provider}/
      judgements.db
      metadata.json

For multi-user sharing, upload the .db to HF Hub (one file per user, or a shared
file with pull-merge-push retry). See push() stub for the HF sync pattern.
"""

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from judgearena.log import get_logger

logger = get_logger(__name__)


class SQLiteCompletionStore:
    """SQLite-backed store for model completions.

    Usage::

        store = SQLiteCompletionStore(db_path="path/to/completions.db")
        store.save(df, pushed_by="alice")

        missing = store.missing_indices(all_instruction_indices)
        df = store.query()
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS completions (
                    instruction_index INTEGER PRIMARY KEY,
                    completion        TEXT,
                    pushed_by         TEXT,
                    pushed_at         TEXT,
                    run_id            TEXT
                )
            """)
            self._conn.commit()
        return self._conn

    def save(
        self,
        df: pd.DataFrame,
        pushed_by: str,
        run_id: str | None = None,
    ) -> int:
        """Insert or replace completions. Returns number of rows written.

        Args:
            df: DataFrame with ``instruction_index`` and ``completion`` columns.
            pushed_by: Username or job identifier.
            run_id: UUID for this batch; auto-generated if omitted.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())
        conn = self._connect()
        now = datetime.now(UTC).isoformat()
        rows = [
            (int(row["instruction_index"]), row["completion"], pushed_by, now, run_id)
            for _, row in df.iterrows()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO completions "
            "(instruction_index, completion, pushed_by, pushed_at, run_id) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        logger.info("Wrote %d completions to %s", len(rows), self.db_path)
        return len(rows)

    def query(self, indices: list[int] | None = None) -> pd.DataFrame:
        """Return completions, optionally filtered to specific instruction_indices."""
        conn = self._connect()
        if indices is None:
            return pd.read_sql(
                "SELECT * FROM completions ORDER BY instruction_index", conn
            )
        placeholders = ",".join("?" * len(indices))
        return pd.read_sql(
            f"SELECT * FROM completions WHERE instruction_index IN ({placeholders})"
            " ORDER BY instruction_index",
            conn,
            params=indices,
        )

    def missing_indices(self, all_indices: list[int]) -> list[int]:
        """Return indices from all_indices not yet in the store."""
        conn = self._connect()
        existing = set(
            pd.read_sql("SELECT instruction_index FROM completions", conn)[
                "instruction_index"
            ].astype(int)
        )
        return [i for i in all_indices if i not in existing]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SQLiteJudgementStore:
    """SQLite-backed store for pairwise LLM judge annotations.

    Each row has its own model_A and model_B string identifiers.
    The unique key is (instruction_index, model_A, model_B).

    Usage::

        store = SQLiteJudgementStore(db_path="path/to/judgements.db")
        # df columns: instruction_index, model_A, model_B, judge_input, judge_output
        store.save(df, pushed_by="alice")

        missing = store.missing_indices(all_indices, model_A="gpt-4o", model_B="my-model")
        df = store.query(model="my-model")
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS judgements (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    instruction_index INTEGER NOT NULL,
                    model_A           TEXT    NOT NULL,
                    model_B           TEXT    NOT NULL,
                    judge_input       TEXT,
                    judge_output      TEXT,
                    pushed_by         TEXT,
                    pushed_at         TEXT,
                    run_id            TEXT,
                    UNIQUE(instruction_index, model_A, model_B)
                )
            """)
            self._conn.commit()
        return self._conn

    def save(
        self,
        df: pd.DataFrame,
        pushed_by: str,
        run_id: str | None = None,
    ) -> int:
        """Insert or replace judgements. Returns number of rows written.

        Args:
            df: DataFrame with ``instruction_index``, ``model_A``, ``model_B``,
                ``judge_output`` columns. ``judge_input`` is optional.
            pushed_by: Username or job identifier.
            run_id: UUID for this batch; auto-generated if omitted.
        """
        required = {"instruction_index", "model_A", "model_B", "judge_output"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"DataFrame missing columns: {missing_cols}")

        if run_id is None:
            run_id = str(uuid.uuid4())
        conn = self._connect()
        now = datetime.now(UTC).isoformat()
        rows = [
            (
                int(row["instruction_index"]),
                row["model_A"],
                row["model_B"],
                row.get("judge_input") if "judge_input" in df.columns else None,
                row["judge_output"],
                pushed_by,
                now,
                run_id,
            )
            for _, row in df.iterrows()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO judgements "
            "(instruction_index, model_A, model_B, judge_input, judge_output, "
            "pushed_by, pushed_at, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        logger.info("Wrote %d judgements to %s", len(rows), self.db_path)
        return len(rows)

    def query(
        self,
        model_A: str | None = None,
        model_B: str | None = None,
        model: str | None = None,
    ) -> pd.DataFrame:
        """Return judgements, optionally filtered by model position.

        Args:
            model_A: Keep only rows where position A is this model.
            model_B: Keep only rows where position B is this model.
            model: Keep rows where this model appears in either position.
                   Cannot be combined with model_A / model_B.
        """
        if model is not None and (model_A is not None or model_B is not None):
            raise ValueError("Use either `model` or `model_A`/`model_B`, not both.")

        conn = self._connect()
        if model is not None:
            return pd.read_sql(
                "SELECT * FROM judgements WHERE model_A = ? OR model_B = ?"
                " ORDER BY instruction_index",
                conn,
                params=(model, model),
            )
        conditions, params = [], []
        if model_A is not None:
            conditions.append("model_A = ?")
            params.append(model_A)
        if model_B is not None:
            conditions.append("model_B = ?")
            params.append(model_B)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return pd.read_sql(
            f"SELECT * FROM judgements {where} ORDER BY instruction_index",
            conn,
            params=params,
        )

    def missing_indices(
        self, all_indices: list[int], model_A: str, model_B: str
    ) -> list[int]:
        """Return indices not yet judged for the given (model_A, model_B) pair."""
        conn = self._connect()
        existing = set(
            pd.read_sql(
                "SELECT instruction_index FROM judgements WHERE model_A = ? AND model_B = ?",
                conn,
                params=(model_A, model_B),
            )["instruction_index"].astype(int)
        )
        return [i for i in all_indices if i not in existing]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
