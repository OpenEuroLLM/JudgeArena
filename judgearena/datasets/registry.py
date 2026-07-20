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
    from judgearena.datasets import (
        arena_hard,
        fluency,
        judgearena_tables,
        m_arenahard,
        mt_bench,
    )

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
        DatasetAdapter(
            "fluency",
            fluency.download_task_sources,
            fluency.load_task_instructions,
            fluency.load_task_model_outputs,
        ),
        DatasetAdapter(
            "m_arena_hard",
            m_arenahard.download_task_sources,
            m_arenahard.load_task_instructions,
            m_arenahard.load_task_model_outputs,
        ),
        DatasetAdapter(
            "mt_bench",
            mt_bench.download_task_sources,
            mt_bench.load_task_instructions,
            mt_bench.load_task_model_outputs,
        ),
    )


def resolve_dataset_adapter(name: str) -> DatasetAdapter:
    """Return the implementation registered under ``name``."""
    for adapter in dataset_adapters():
        if adapter.name == name:
            return adapter
    raise ValueError(f"Unknown task dataset adapter {name!r}.")
