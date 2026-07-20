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
    """Route explicitly named or YAML-selected tasks to one runner."""

    name: str
    tasks: frozenset[str]
    runner: BenchmarkRunner

    def supports(self, task: str) -> bool:
        return task in self.tasks


@dataclass(frozen=True)
class ResolvedBenchmark:
    """Runner selection and the task definition resolved during dispatch."""

    adapter: BenchmarkAdapter
    task: ResolvedTaskSpec | None


def benchmark_adapters() -> tuple[BenchmarkAdapter, ...]:
    """Return registered benchmark implementations, specific first."""
    from judgearena.benchmarks.mt_bench.runner import run_mt_bench_benchmark
    from judgearena.benchmarks.pairwise.runner import run_pairwise

    return (
        BenchmarkAdapter("mt_bench", frozenset(), run_mt_bench_benchmark),
        BenchmarkAdapter("pairwise", frozenset(), run_pairwise),
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
