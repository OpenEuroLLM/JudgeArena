"""LMArena human preference datasets."""

from pathlib import Path

import pandas as pd

from judgearena.datasets._arena import (
    _download_arena_dataset,
    arena_sources,
    download_arena_sources,
    normalize_battles,
    single_turn_battles,
)
from judgearena.datasets.data import PreferenceData
from judgearena.datasets.registry import DatasetAdapter

_RELEASES = ("LMArena-100k", "LMArena-55k", "LMArena-140k")


def _read_release(arena, dataset_sources):
    if arena == "LMArena-55k":
        repo_id = "lmarena-ai/arena-human-preference-55k"
        path = _download_arena_dataset(
            repo_id=repo_id,
            default_allow_patterns="*.csv",
            dataset_sources=dataset_sources,
        )
        df = pd.read_csv(Path(path) / "train.csv")

        def _winner_55k(row) -> str | None:
            if row["winner_tie"]:
                return "tie"
            if row["winner_model_a"]:
                return "model_a"
            if row["winner_model_b"]:
                return "model_b"
            return None

        df["winner"] = df.apply(_winner_55k, axis=1)
        df = df[df["winner"].notna()].copy()

        df["conversation_a"] = df.apply(
            lambda r: [
                {"role": "user", "content": str(r["prompt"])},
                {"role": "assistant", "content": str(r["response_a"])},
            ],
            axis=1,
        )
        df["conversation_b"] = df.apply(
            lambda r: [
                {"role": "user", "content": str(r["prompt"])},
                {"role": "assistant", "content": str(r["response_b"])},
            ],
            axis=1,
        )
        df["question_id"] = df["id"]
        df["tstamp"] = 0
        df["benchmark"] = "LMArena-55k"

    else:
        size = arena.split("-")[1]  # "100k" or "140k"
        repo_id = f"lmarena-ai/arena-human-preference-{size}"
        path = _download_arena_dataset(
            repo_id=repo_id,
            default_allow_patterns="*parquet",
            dataset_sources=dataset_sources,
        )
        parquet_files = sorted((Path(path) / "data").glob("*.parquet"))
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

        if "tstamp" in df.columns:
            # 100k: tstamp is a unix timestamp in seconds
            df["date"] = pd.to_datetime(df["tstamp"], unit="s")
        else:
            # 140k: timestamp is already a datetime
            df["tstamp"] = df["timestamp"].astype("int64") // 10**9
            df["date"] = df["timestamp"]

        if "question_id" not in df.columns:
            df["question_id"] = df["id"]

        # 140k uses "both_bad" instead of "tie (bothbad)"
        df["winner"] = df["winner"].replace("both_bad", "tie (bothbad)")

        df["benchmark"] = arena

    return single_turn_battles(df, arena)


class LMArenaAdapter(DatasetAdapter):
    def download(self) -> None:
        download_arena_sources(self.task)

    def load_data(self) -> PreferenceData:
        arena = self.task.spec.dataset.format or "LMArena"
        sources = arena_sources(self.task)
        if arena == "LMArena":
            battles = pd.concat(
                [_read_release(name, sources) for name in _RELEASES], ignore_index=True
            )
        elif arena in _RELEASES:
            battles = _read_release(arena, sources)
        else:
            raise ValueError(f"Unsupported LMArena format {arena!r}.")
        return normalize_battles(
            battles,
            arena,
            outcome_preferences=self.task.spec.dataset.outcome_preferences,
        )
