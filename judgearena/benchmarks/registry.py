"""Registry of benchmark task names and their runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from judgearena.tasks.registry import get_packaged_task

if TYPE_CHECKING:
    from judgearena.config import RunConfig


class BenchmarkRunner(Protocol):
    """Callable implemented by a benchmark-specific evaluation module."""

    def __call__(self, cfg: RunConfig, /) -> object: ...


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


def benchmark_adapters() -> tuple[BenchmarkAdapter, ...]:
    """Return the registered benchmark implementations."""
    from judgearena.benchmarks.pairwise.runner import run_pairwise

    return (BenchmarkAdapter("pairwise", None, run_pairwise),)


def resolve_benchmark_adapter(task: str) -> BenchmarkAdapter:
    """Resolve a YAML-selected runner, then fall back for unmigrated tasks."""
    adapters = benchmark_adapters()
    resolved = get_packaged_task(task)
    if resolved is not None:
        runner_id = resolved.spec.protocol.runner
        for adapter in adapters:
            if adapter.name == runner_id:
                return adapter
        raise ValueError(
            f"Task {task!r} selects unavailable runner {runner_id!r}."
        )

    for adapter in adapters:
        if adapter.supports(task):
            return adapter
    raise ValueError(f"No generate-and-evaluate adapter supports task {task!r}.")
