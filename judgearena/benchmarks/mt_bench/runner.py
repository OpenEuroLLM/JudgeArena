"""Registered entry point for the specialized MT-Bench pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from judgearena.artifacts import prepare_run_directory
from judgearena.benchmarks.mt_bench.mt_bench_utils import run_mt_bench
from judgearena.benchmarks.pairwise.baselines import native_pairwise_baseline

if TYPE_CHECKING:
    from judgearena.config import RunConfig


def run_mt_bench_benchmark(cfg: RunConfig):
    """Prepare one run directory and execute the YAML-selected MT-Bench runner."""
    baseline = cfg.model.baseline or native_pairwise_baseline(cfg.task)
    if not isinstance(baseline, str):
        raise ValueError("MT-Bench requires a flat native baseline.")

    result_name = (
        f"{cfg.task}-{cfg.model.name}-{baseline}-{cfg.judge.model}-"
        f"{cfg.judge.swap_mode}"
    ).replace("/", "_")
    run_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    result_folder = prepare_run_directory(
        cfg,
        Path(cfg.run.result_folder) / f"{result_name}-{run_timestamp}",
    )
    return run_mt_bench(
        cfg,
        cfg.run.ignore_cache,
        res_folder=result_folder,
        result_name=result_name,
    )
