"""Unified dispatcher for benchmark generation and evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from judgearena.benchmarks.registry import resolve_benchmark
from judgearena.datasets.data import CompletionData
from judgearena.log import get_logger
from judgearena.usage import track_usage

if TYPE_CHECKING:
    from judgearena.config import RunConfig

logger = get_logger(__name__)


def run_benchmark(
    cfg: RunConfig,
    *,
    data: CompletionData | None = None,
    output_dir=None,
    **runner_options,
) -> object:
    """Run a task; an explicit output directory returns its evaluation manifest.

    Runner-specific preparation inputs (such as a fixed baseline plan) can be
    supplied by Python callers without changing the task's judging/scoring settings.
    """
    if data is not None and not isinstance(data, CompletionData):
        raise TypeError("Supplied data must be CompletionData or PreferenceData.")
    if (
        data is None
        and cfg.resolved_task is not None
        and cfg.resolved_task.spec.dataset.adapter == "supplied"
    ):
        raise ValueError(
            "This evaluation requires supplied data: run_benchmark(cfg, data=data)."
        )
    resolved = resolve_benchmark(cfg.resolved_task)
    logger.info("Using %s benchmark adapter for %s.", resolved.adapter.name, cfg.task)
    with track_usage() as usage_tracker:
        try:
            return resolved.adapter.runner(
                cfg,
                resolved.task,
                **({"data": data} if data is not None else {}),
                **({"output_dir": output_dir} if output_dir is not None else {}),
                **runner_options,
            )
        finally:
            usage_tracker.render_summary()
