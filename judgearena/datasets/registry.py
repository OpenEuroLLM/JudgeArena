"""Registry connecting task dataset-adapter IDs to implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from judgearena.tasks.schema import ResolvedTaskSpec

TaskDataFunction = Callable[["ResolvedTaskSpec", Path], pd.DataFrame | None]
TaskDownloadFunction = Callable[["ResolvedTaskSpec", Path], None]
TaskBattleFunction = Callable[["ResolvedTaskSpec", Path], pd.DataFrame]


@dataclass(frozen=True)
class InstructionDatasetAdapter:
    """Dataset operations required by instruction-based benchmark runners."""

    name: str
    download: TaskDownloadFunction
    load_instructions: TaskDataFunction
    load_model_outputs: TaskDataFunction


@dataclass(frozen=True)
class BattleDatasetAdapter:
    """Dataset operations required by arena-battle benchmark runners."""

    name: str
    download: TaskDownloadFunction
    load_battles: TaskBattleFunction


TaskDatasetAdapter = InstructionDatasetAdapter | BattleDatasetAdapter


@dataclass(frozen=True)
class _DatasetRegistration:
    """Lazy dataset registration kept import-safe for task validation."""

    name: str
    module: str
    kind: Literal["instructions", "battles"] = "instructions"


_DATASET_REGISTRATIONS = (
    _DatasetRegistration(
        "arena_battles", "judgearena.datasets.arena_battles", "battles"
    ),
    _DatasetRegistration("judgearena_tables", "judgearena.datasets.judgearena_tables"),
    _DatasetRegistration("arena_hard", "judgearena.datasets.arena_hard"),
    _DatasetRegistration("fluency", "judgearena.datasets.fluency"),
    _DatasetRegistration("m_arena_hard", "judgearena.datasets.m_arenahard"),
    _DatasetRegistration("mt_bench", "judgearena.datasets.mt_bench"),
    _DatasetRegistration("wildbench", "judgearena.datasets.wildbench"),
)

DATASET_ADAPTER_NAMES = frozenset(
    registration.name for registration in _DATASET_REGISTRATIONS
)
INSTRUCTION_DATASET_ADAPTER_NAMES = frozenset(
    registration.name
    for registration in _DATASET_REGISTRATIONS
    if registration.kind == "instructions"
)
BATTLE_DATASET_ADAPTER_NAMES = frozenset(
    registration.name
    for registration in _DATASET_REGISTRATIONS
    if registration.kind == "battles"
)


def _build_adapter(registration: _DatasetRegistration) -> TaskDatasetAdapter:
    module = import_module(registration.module)
    if registration.kind == "battles":
        return BattleDatasetAdapter(
            registration.name,
            module.download_task_sources,
            module.load_task_battles,
        )
    return InstructionDatasetAdapter(
        registration.name,
        module.download_task_sources,
        module.load_task_instructions,
        module.load_task_model_outputs,
    )


def dataset_adapters() -> tuple[TaskDatasetAdapter, ...]:
    """Return registered dataset implementations."""
    return tuple(_build_adapter(item) for item in _DATASET_REGISTRATIONS)


def resolve_dataset_adapter(name: str) -> InstructionDatasetAdapter:
    """Return the instruction-dataset implementation registered under ``name``."""
    for adapter in dataset_adapters():
        if isinstance(adapter, InstructionDatasetAdapter) and adapter.name == name:
            return adapter
    raise ValueError(f"Unknown instruction dataset adapter {name!r}.")


def resolve_battle_dataset_adapter(name: str) -> BattleDatasetAdapter:
    """Return the battle-dataset implementation registered under ``name``."""
    for adapter in dataset_adapters():
        if isinstance(adapter, BattleDatasetAdapter) and adapter.name == name:
            return adapter
    raise ValueError(f"Unknown battle dataset adapter {name!r}.")


def resolve_download_adapter(name: str) -> TaskDatasetAdapter:
    """Return any registered dataset adapter for source prefetching."""
    for adapter in dataset_adapters():
        if adapter.name == name:
            return adapter
    raise ValueError(f"Unknown task dataset adapter {name!r}.")
