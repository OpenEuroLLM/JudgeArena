"""Registry connecting task dataset-adapter IDs to implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from judgearena.tasks.schema import ResolvedTaskSpec

TaskDataFunction = Callable[["ResolvedTaskSpec", Path], pd.DataFrame | None]
TaskDownloadFunction = Callable[["ResolvedTaskSpec", Path], None]


@dataclass(frozen=True)
class DatasetAdapter:
    name: str
    download: TaskDownloadFunction
    load_instructions: TaskDataFunction
    load_model_outputs: TaskDataFunction


@dataclass(frozen=True)
class _DatasetRegistration:
    """Lazy dataset registration kept import-safe for task validation."""

    name: str
    module: str


_DATASET_REGISTRATIONS = (
    _DatasetRegistration("judgearena_tables", "judgearena.datasets.judgearena_tables"),
    _DatasetRegistration("arena_hard", "judgearena.datasets.arena_hard"),
    _DatasetRegistration("fluency", "judgearena.datasets.fluency"),
    _DatasetRegistration("m_arena_hard", "judgearena.datasets.m_arenahard"),
    _DatasetRegistration("mt_bench", "judgearena.datasets.mt_bench"),
)

DATASET_ADAPTER_NAMES = frozenset(
    registration.name for registration in _DATASET_REGISTRATIONS
)


def _build_adapter(registration: _DatasetRegistration) -> DatasetAdapter:
    module = import_module(registration.module)
    return DatasetAdapter(
        registration.name,
        module.download_task_sources,
        module.load_task_instructions,
        module.load_task_model_outputs,
    )


def dataset_adapters() -> tuple[DatasetAdapter, ...]:
    """Return registered dataset implementations."""
    return tuple(_build_adapter(item) for item in _DATASET_REGISTRATIONS)


def resolve_dataset_adapter(name: str) -> DatasetAdapter:
    """Return the implementation registered under ``name``."""
    for adapter in dataset_adapters():
        if adapter.name == name:
            return adapter
    raise ValueError(f"Unknown task dataset adapter {name!r}.")
