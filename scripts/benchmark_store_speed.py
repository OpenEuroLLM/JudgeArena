"""Benchmark SQLiteCompletionStore / SQLiteJudgementStore load and missing_indices speed.

Usage:
    uv run python scripts/benchmark_store_speed.py
    uv run python scripts/benchmark_store_speed.py --db-root ~/judgearena-data/db --n-sample 1000
"""

import argparse
import random
import time
from pathlib import Path

import pandas as pd

from judgearena.store_sqlite import SQLiteCompletionStore, SQLiteJudgementStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-root",
        type=str,
        default="~/judgearena-data/db",
        help="Root folder containing completions/ and judgements/ subdirs.",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=1000,
        help="Number of random instruction indices to probe with missing_indices().",
    )
    return parser.parse_args()


def benchmark_completions(db_path: Path, n_sample: int) -> None:
    t0 = time.perf_counter()
    store = SQLiteCompletionStore(db_path)
    store._connect()
    t1 = time.perf_counter()

    n_rows = pd.read_sql("SELECT COUNT(*) as n FROM completions", store._conn).iloc[0][
        "n"
    ]
    all_indices = list(range(int(n_rows) * 2))  # half present, half not

    sample = random.sample(all_indices, min(n_sample, len(all_indices)))
    t2 = time.perf_counter()
    missing = store.missing_indices(sample)
    t3 = time.perf_counter()

    store.close()

    print(f"[completions] {db_path}")
    print(f"  rows:                 {int(n_rows)}")
    print(f"  load (connect):       {(t1 - t0) * 1000:.2f} ms")
    print(
        f"  missing_indices({len(sample)}): {(t3 - t2) * 1000:.2f} ms  ({len(missing)} missing)"
    )


def benchmark_judgements(db_path: Path, n_sample: int) -> None:
    t0 = time.perf_counter()
    store = SQLiteJudgementStore(db_path)
    store._connect()
    t1 = time.perf_counter()

    n_rows = pd.read_sql("SELECT COUNT(*) as n FROM judgements", store._conn).iloc[0][
        "n"
    ]
    pair = pd.read_sql("SELECT model_A, model_B FROM judgements LIMIT 1", store._conn)
    if pair.empty:
        store.close()
        print(f"[judgements] {db_path} — empty, skipping")
        return
    model_a, model_b = pair.iloc[0]["model_A"], pair.iloc[0]["model_B"]

    all_indices = list(range(int(n_rows) * 2))  # half present, half not
    sample = random.sample(all_indices, min(n_sample, len(all_indices)))

    t2 = time.perf_counter()
    missing = store.missing_indices(sample, model_A=model_a, model_B=model_b)
    t3 = time.perf_counter()

    store.close()

    print(f"[judgements] {db_path}")
    print(f"  rows:                 {int(n_rows)}")
    print(f"  load (connect):       {(t1 - t0) * 1000:.2f} ms")
    print(
        f"  missing_indices({len(sample)}): {(t3 - t2) * 1000:.2f} ms  ({len(missing)} missing)"
    )


def main() -> None:
    args = parse_args()
    db_root = Path(args.db_root).expanduser()

    completion_dbs = sorted((db_root / "completions").glob("**/completions.db"))
    judgement_dbs = sorted((db_root / "judgements").glob("**/judgements.db"))

    if not completion_dbs and not judgement_dbs:
        raise SystemExit(f"No .db files found under {db_root}")

    for db_path in completion_dbs:
        benchmark_completions(db_path, args.n_sample)
        print()

    for db_path in judgement_dbs:
        benchmark_judgements(db_path, args.n_sample)
        print()


if __name__ == "__main__":
    main()
