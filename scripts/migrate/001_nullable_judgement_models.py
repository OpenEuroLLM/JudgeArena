"""Allow judgments without model attribution in existing caches; run with writers stopped."""

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

COLUMNS = (
    "input_hash, judge_input, judge_completion, benchmark, instruction_id, "
    "model_a, model_b, judge, top_logprobs, orientation, pushed_by, pushed_at, run_id"
)
SCHEMA = """
    CREATE TABLE judgements (
        input_hash TEXT PRIMARY KEY, judge_input TEXT NOT NULL,
        judge_completion TEXT NOT NULL, benchmark TEXT NOT NULL,
        instruction_id TEXT NOT NULL, model_a TEXT, model_b TEXT,
        judge TEXT NOT NULL, top_logprobs TEXT, orientation TEXT,
        pushed_by TEXT NOT NULL, pushed_at TEXT NOT NULL, run_id TEXT NOT NULL
    )
"""


def migrate(path: Path) -> bool:
    """Back up and migrate one database; return False when already migrated."""
    with closing(
        sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True)
    ) as conn:
        columns = {row[1]: row for row in conn.execute("PRAGMA table_info(judgements)")}
        if set(columns) != set(COLUMNS.split(", ")):
            raise ValueError(f"Unrecognized judgment-cache schema: {path}")
        if not columns["model_a"][3]:
            return False
        backup_path = Path(str(path) + ".bak")
        with backup_path.open("xb"):
            pass
        with closing(sqlite3.connect(backup_path)) as backup:
            conn.backup(backup)
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ALTER TABLE judgements RENAME TO judgements_with_models")
            conn.execute(SCHEMA)
            conn.execute(
                f"INSERT INTO judgements ({COLUMNS}) "
                f"SELECT {COLUMNS} FROM judgements_with_models"
            )
            conn.execute("DROP TABLE judgements_with_models")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Judgment database or cache directory")
    args = parser.parse_args()
    if not args.path.exists():
        parser.error(f"Path does not exist: {args.path}")
    paths = (
        [args.path] if args.path.is_file() else sorted(args.path.rglob("judgements.db"))
    )
    for path in paths:
        changed = migrate(path)
        print(f"{'Migrated' if changed else 'Already migrated'}: {path}")


if __name__ == "__main__":
    main()
