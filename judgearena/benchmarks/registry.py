"""Registry of benchmark task names and their runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import ResolvedTaskSpec

if TYPE_CHECKING:
    from judgearena.config import RunConfig


class BenchmarkRunner(Protocol):
    """Callable implemented by a benchmark-specific evaluation module."""

    def __call__(self, cfg: RunConfig, task: ResolvedTaskSpec | None, /) -> object: ...


@dataclass(frozen=True)
class BenchmarkAdapter:
    """Route a set of task names to one benchmark runner.

    ``tasks=None`` defines the fallback adapter and should therefore be last.
    Adding a benchmark only requires its runner and one registry entry.
    """

    name: str
    tasks: frozenset[str] | None
    runner: BenchmarkRunner

    def supports(self, task: str) -> bool:
        return self.tasks is None or task in self.tasks


@dataclass(frozen=True)
class ResolvedBenchmark:
    """Runner selection and the task definition resolved during dispatch."""

    adapter: BenchmarkAdapter
    task: ResolvedTaskSpec | None


def benchmark_adapters() -> tuple[BenchmarkAdapter, ...]:
    """Return registered benchmark implementations, specific first."""
    from judgearena.benchmarks.elo.runner import run_elo
    from judgearena.benchmarks.meta_eval.runner import run_meta_eval
    from judgearena.benchmarks.mt_bench.runner import run_mt_bench_benchmark
    from judgearena.benchmarks.pairwise.runner import run_pairwise

    return (
        BenchmarkAdapter("elo", frozenset(), run_elo),
        BenchmarkAdapter("meta_eval", frozenset(), run_meta_eval),
        BenchmarkAdapter("mt_bench", frozenset(), run_mt_bench_benchmark),
        BenchmarkAdapter("pairwise", None, run_pairwise),
    )


def resolve_benchmark(task: str) -> ResolvedBenchmark:
    """Resolve a runner and task definition with one registry lookup."""
    adapters = benchmark_adapters()
    resolved = get_packaged_task(task)
    if resolved is not None:
        runner_id = resolved.spec.protocol.runner
        for adapter in adapters:
            if adapter.name == runner_id:
                return ResolvedBenchmark(adapter=adapter, task=resolved)
        raise ValueError(f"Task {task!r} selects unavailable runner {runner_id!r}.")

    for adapter in adapters:
        if adapter.supports(task):
            return ResolvedBenchmark(adapter=adapter, task=None)
    raise ValueError(f"No generate-and-evaluate adapter supports task {task!r}.")


def resolve_benchmark_adapter(task: str) -> BenchmarkAdapter:
    """Return only the selected adapter for compatibility and inspection."""
    return resolve_benchmark(task).adapter
