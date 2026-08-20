"""Unified dispatcher for benchmark generation and evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from judgearena.benchmarks.registry import resolve_benchmark_adapter
from judgearena.log import get_logger

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


def run_benchmark(cfg: RunConfig) -> object:
    """Run a task through its registered benchmark adapter."""
    adapter = resolve_benchmark_adapter(cfg.task)
    logger.info("Using %s benchmark adapter for %s.", adapter.name, cfg.task)
    return adapter.runner(cfg)
