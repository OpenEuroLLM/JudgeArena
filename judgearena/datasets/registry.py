"""Registry connecting task dataset-adapter IDs to implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from judgearena.tasks.schema import ResolvedTaskSpec

TaskDataFunction = Callable[[ResolvedTaskSpec, Path], pd.DataFrame | None]
TaskDownloadFunction = Callable[[ResolvedTaskSpec, Path], None]


@dataclass(frozen=True)
class DatasetAdapter:
    name: str
    download: TaskDownloadFunction
    load_instructions: TaskDataFunction
    load_model_outputs: TaskDataFunction


def dataset_adapters() -> tuple[DatasetAdapter, ...]:
    """Return registered dataset implementations."""
    from judgearena.datasets import arena_hard, judgearena_tables

    return (
        DatasetAdapter(
            "judgearena_tables",
            judgearena_tables.download_task_sources,
            judgearena_tables.load_task_instructions,
            judgearena_tables.load_task_model_outputs,
        ),
        DatasetAdapter(
            "arena_hard",
            arena_hard.download_task_sources,
            arena_hard.load_task_instructions,
            arena_hard.load_task_model_outputs,
        ),
    )


def resolve_dataset_adapter(name: str) -> DatasetAdapter:
    """Return the implementation registered under ``name``."""
    for adapter in dataset_adapters():
        if adapter.name == name:
            return adapter
    raise ValueError(f"Unknown task dataset adapter {name!r}.")
