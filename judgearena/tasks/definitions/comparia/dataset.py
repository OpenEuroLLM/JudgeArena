"""ComparIA human preference votes."""

import warnings
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


class ComparIAAdapter(DatasetAdapter):
    def download(self) -> None:
        download_arena_sources(self.task)

    def load_data(self) -> PreferenceData:
        path = _download_arena_dataset(
            repo_id="ministere-culture/comparia-votes",
            default_allow_patterns="*",
            dataset_sources=arena_sources(self.task),
        )

        df = pd.read_parquet(Path(path) / "votes.parquet")

        df["tstamp"] = df["timestamp"]
        df["model_a"] = df["model_a_name"]
        df["model_b"] = df["model_b_name"]

        def get_winner(
            chosen_model_name: str,
            model_a: str,
            model_b: str,
            both_equal: bool,
            **kwargs,
        ):
            if both_equal:
                return "tie"
            else:
                if chosen_model_name is None or isinstance(chosen_model_name, float):
                    return None
                if chosen_model_name not in [model_a, model_b]:
                    warnings.warn(
                        f"Chosen model {chosen_model_name!r} not in model_a={model_a!r} or model_b={model_b!r}; skipping.",
                        stacklevel=2,
                    )
                    return None
                return "model_a" if chosen_model_name == model_a else "model_b"

        df["winner"] = df.apply(lambda row: get_winner(**row), axis=1)

        df = df[~df.winner.isna()]
        df["benchmark"] = "ComparIA"
        df["question_id"] = df["id"]

        return normalize_battles(
            single_turn_battles(df, "ComparIA"),
            "ComparIA",
            outcome_preferences=self.task.spec.dataset.outcome_preferences,
        )
