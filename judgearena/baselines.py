"""Dataset-native baselines for pairwise generate-and-evaluate benchmarks."""

from __future__ import annotations

from collections.abc import Mapping

from judgearena.instruction_dataset.arena_hard import ARENA_HARD_BASELINES
from judgearena.instruction_dataset.m_arenahard import (
    M_ARENA_HARD_BASELINES,
    split_m_arena_hard_dataset,
)
from judgearena.instruction_dataset.mt_bench import MT_BENCH_BASELINES

ALPACA_EVAL_BASELINES: dict[str, str] = {
    "alpaca-eval": "gpt4_1106_preview",
}

PAIRWISE_BASELINES: dict[str, str | Mapping[str, str]] = {
    **ALPACA_EVAL_BASELINES,
    **ARENA_HARD_BASELINES,
    **M_ARENA_HARD_BASELINES,
    **MT_BENCH_BASELINES,
}


def native_pairwise_baseline(task: str) -> str | Mapping[str, str] | None:
    """Return the dataset-native pairwise baseline, if the task defines one."""
    if task in PAIRWISE_BASELINES:
        return PAIRWISE_BASELINES[task]
    parsed_m_arena_hard = split_m_arena_hard_dataset(task)
    if parsed_m_arena_hard is not None:
        version_key, _lang_or_subset = parsed_m_arena_hard
        return PAIRWISE_BASELINES[version_key]
    return None
