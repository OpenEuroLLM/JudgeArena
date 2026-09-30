"""Unified dispatcher for benchmark generation and evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from judgearena.benchmarks.registry import resolve_benchmark
from judgearena.log import get_logger
from judgearena.usage import track_usage

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


def run_benchmark(cfg: RunConfig) -> object:
    """Run a task through its registered benchmark adapter."""
    resolved = resolve_benchmark(cfg.task)
    logger.info("Using %s benchmark adapter for %s.", resolved.adapter.name, cfg.task)
    with track_usage() as usage_tracker:
        try:
            if cfg.tune_judge is not None:
                from judgearena.tuning.runner import run_tune_judge

                return run_tune_judge(cfg, resolved.task)
            return resolved.adapter.runner(cfg, resolved.task)
        finally:
            usage_tracker.render_summary()
