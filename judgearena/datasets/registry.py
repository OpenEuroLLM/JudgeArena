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

    download: TaskDownloadFunction
    load_instructions: TaskDataFunction
    load_model_outputs: TaskDataFunction


@dataclass(frozen=True)
class BattleDatasetAdapter:
    """Dataset operations required by arena-battle benchmark runners."""

    download: TaskDownloadFunction
    load_battles: TaskBattleFunction


TaskDatasetAdapter = InstructionDatasetAdapter | BattleDatasetAdapter


def _instruction_datasets() -> dict[str, InstructionDatasetAdapter]:
    from judgearena.datasets import (
        alpaca_eval,
        arena_hard,
        fluency,
        judgearena_tables,
        m_arenahard,
        mt_bench,
    )

    return {
        "alpaca_eval": InstructionDatasetAdapter(
            alpaca_eval.download_task_sources,
            alpaca_eval.load_task_instructions,
            alpaca_eval.load_task_model_outputs,
        ),
        "judgearena_tables": InstructionDatasetAdapter(
            judgearena_tables.download_task_sources,
            judgearena_tables.load_task_instructions,
            judgearena_tables.load_task_model_outputs,
        ),
        "arena_hard": InstructionDatasetAdapter(
            arena_hard.download_task_sources,
            arena_hard.load_task_instructions,
            arena_hard.load_task_model_outputs,
        ),
        "fluency": InstructionDatasetAdapter(
            fluency.download_task_sources,
            fluency.load_task_instructions,
            fluency.load_task_model_outputs,
        ),
        "m_arena_hard": InstructionDatasetAdapter(
            m_arenahard.download_task_sources,
            m_arenahard.load_task_instructions,
            m_arenahard.load_task_model_outputs,
        ),
        "mt_bench": InstructionDatasetAdapter(
            mt_bench.download_task_sources,
            mt_bench.load_task_instructions,
            mt_bench.load_task_model_outputs,
        ),
    }


def _battle_datasets() -> dict[str, BattleDatasetAdapter]:
    from judgearena.datasets import arena_battles

    return {
        "arena_battles": BattleDatasetAdapter(
            arena_battles.download_task_sources,
            arena_battles.load_task_battles,
        ),
    }


def resolve_dataset_adapter(name: str) -> InstructionDatasetAdapter:
    """Return the instruction-dataset implementation registered under ``name``."""
    return _instruction_datasets()[name]


def resolve_battle_dataset_adapter(name: str) -> BattleDatasetAdapter:
    """Return the battle-dataset implementation registered under ``name``."""
    return _battle_datasets()[name]


def resolve_download_adapter(name: str) -> TaskDatasetAdapter:
    """Return any registered dataset adapter for source prefetching."""
    return {**_instruction_datasets(), **_battle_datasets()}[name]
