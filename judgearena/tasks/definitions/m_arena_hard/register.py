"""Register this task family's dataset adapter."""

from judgearena.datasets.registry import register_dataset
from judgearena.tasks.definitions.m_arena_hard.dataset import MArenaHardAdapter

register_dataset("m_arena_hard")(MArenaHardAdapter)
