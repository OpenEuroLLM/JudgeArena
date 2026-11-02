"""Registry of benchmark task names and their runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import ResolvedTaskSpec

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.datasets.data import CompletionData


class BenchmarkRunner(Protocol):
    """Callable implemented by a benchmark-specific evaluation module."""

    def __call__(
        self,
        cfg: RunConfig,
        task: ResolvedTaskSpec | None,
        /,
        *,
        data: CompletionData | None = None,
    ) -> object: ...


@dataclass(frozen=True)
class BenchmarkAdapter:
    """Associate a task protocol runner ID with its implementation."""

    name: str
    runner: BenchmarkRunner


@dataclass(frozen=True)
class ResolvedBenchmark:
    """Runner selection and the task definition resolved during dispatch."""

    adapter: BenchmarkAdapter
    task: ResolvedTaskSpec


def benchmark_adapters() -> tuple[BenchmarkAdapter, ...]:
    """Return registered benchmark implementations."""
    from judgearena.benchmarks.elo.runner import run_elo
    from judgearena.benchmarks.evaluation import run_evaluation
    from judgearena.benchmarks.meta_eval.runner import run_meta_eval

    return (
        BenchmarkAdapter("elo", run_elo),
        BenchmarkAdapter("meta_eval", run_meta_eval),
        BenchmarkAdapter("pairwise", run_evaluation),
    )


def benchmark_runner_names() -> frozenset[str]:
    """Return runner IDs owned by the benchmark registry."""
    return frozenset(adapter.name for adapter in benchmark_adapters())


def resolve_benchmark(task: str | ResolvedTaskSpec) -> ResolvedBenchmark:
    """Select the runner declared by a resolved task or packaged task name."""
    resolved = get_packaged_task(task) if isinstance(task, str) else task
    if resolved is None:
        raise ValueError(f"Unknown task {task!r}.")
    runner_id = resolved.spec.protocol.runner
    for adapter in benchmark_adapters():
        if adapter.name == runner_id:
            return ResolvedBenchmark(adapter=adapter, task=resolved)
    raise ValueError(
        f"Task {resolved.task!r} selects unavailable runner {runner_id!r}."
    )
