"""Registry of benchmark task names and their runners."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.tasks.schema import ResolvedTaskSpec


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


@dataclass(frozen=True)
class _BenchmarkRegistration:
    """Lazy runner registration kept import-safe for task validation."""

    name: str
    tasks: frozenset[str]
    module: str
    function: str


_BENCHMARK_REGISTRATIONS = (
    _BenchmarkRegistration(
        "elo",
        frozenset(),
        "judgearena.benchmarks.elo.runner",
        "run_elo",
    ),
    _BenchmarkRegistration(
        "mt_bench",
        frozenset(),
        "judgearena.benchmarks.mt_bench.runner",
        "run_mt_bench_benchmark",
    ),
    _BenchmarkRegistration(
        "pairwise",
        frozenset(),
        "judgearena.benchmarks.pairwise.runner",
        "run_pairwise",
    ),
)

BENCHMARK_ADAPTER_NAMES = frozenset(
    registration.name for registration in _BENCHMARK_REGISTRATIONS
)


def benchmark_adapters() -> tuple[BenchmarkAdapter, ...]:
    """Return registered benchmark implementations, specific first."""
    return tuple(
        BenchmarkAdapter(
            registration.name,
            registration.tasks,
            getattr(import_module(registration.module), registration.function),
        )
        for registration in _BENCHMARK_REGISTRATIONS
    )


def get_packaged_task(task: str) -> ResolvedTaskSpec | None:
    """Resolve lazily so task validation can import adapter names safely."""
    from judgearena.tasks.registry import get_packaged_task as lookup

    return lookup(task)


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
