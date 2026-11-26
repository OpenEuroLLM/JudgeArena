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

# Preserve the row order of combined releases for seeded sampling and replay.
_RELEASES = {
    f"lmarena-ai/arena-human-preference-{size}": f"LMArena-{size}"
    for size in ("100k", "55k", "140k")
}


def _read_release(repo_id, dataset_sources):
    arena = _RELEASES[repo_id]
    if arena == "LMArena-55k":
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
    def _sources(self):
        sources = arena_sources(self.task)
        unknown = sources.keys() - _RELEASES.keys()
        if unknown:
            raise ValueError(f"Unsupported LMArena repositories: {sorted(unknown)}")
        return sources

    def download(self) -> None:
        download_arena_sources(self._sources())

    def load_data(self) -> PreferenceData:
        sources = self._sources()
        releases = [repo for repo in _RELEASES if repo in sources]
        battles = pd.concat(
            [_read_release(repo, sources) for repo in releases], ignore_index=True
        )
        arena = _RELEASES[releases[0]] if len(releases) == 1 else "LMArena"
        return normalize_battles(
            battles,
            arena,
            outcome_preferences=self.task.spec.dataset.outcome_preferences,
        )
