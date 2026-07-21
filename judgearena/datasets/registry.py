"""Registry connecting task dataset-adapter IDs to implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from judgearena.tasks.schema import ResolvedTaskSpec

TaskDataFunction = Callable[[ResolvedTaskSpec, Path], pd.DataFrame | None]
TaskDownloadFunction = Callable[[ResolvedTaskSpec, Path], None]
TaskBattleFunction = Callable[[ResolvedTaskSpec, Path], pd.DataFrame]


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


def dataset_adapters() -> tuple[TaskDatasetAdapter, ...]:
    """Return registered dataset implementations."""
    from judgearena.datasets import (
        arena_battles,
        arena_hard,
        judgearena_tables,
        m_arenahard,
        mt_bench,
    )

    return (
        BattleDatasetAdapter(
            "arena_battles",
            arena_battles.download_task_sources,
            arena_battles.load_task_battles,
        ),
        InstructionDatasetAdapter(
            "judgearena_tables",
            judgearena_tables.download_task_sources,
            judgearena_tables.load_task_instructions,
            judgearena_tables.load_task_model_outputs,
        ),
        InstructionDatasetAdapter(
            "arena_hard",
            arena_hard.download_task_sources,
            arena_hard.load_task_instructions,
            arena_hard.load_task_model_outputs,
        ),
        InstructionDatasetAdapter(
            "m_arena_hard",
            m_arenahard.download_task_sources,
            m_arenahard.load_task_instructions,
            m_arenahard.load_task_model_outputs,
        ),
        InstructionDatasetAdapter(
            "mt_bench",
            mt_bench.download_task_sources,
            mt_bench.load_task_instructions,
            mt_bench.load_task_model_outputs,
        ),
    )


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
