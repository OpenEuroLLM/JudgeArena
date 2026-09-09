"""Registered MT-Bench-101 runner.

Evaluation (generation, 1-10 judging, min-per-dialogue aggregation) lands in
the follow-up branch. This stub exists so task YAML can reference
``runner: mt_bench_101``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.tasks.schema import ResolvedTaskSpec


def run_mt_bench_101_benchmark(
    cfg: RunConfig, task: ResolvedTaskSpec | None = None
) -> object:
    task_id = task.task if task is not None else cfg.task
    raise NotImplementedError(
        "MT-Bench-101 evaluation is not implemented in this stack layer. "
        f"Task {task_id!r} is declared and can load instructions."
    )
