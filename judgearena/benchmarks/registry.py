"""Task-to-runner registry for generate-and-evaluate benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

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


def resolve_benchmark_adapter(
    task: str, adapters: tuple[BenchmarkAdapter, ...]
) -> BenchmarkAdapter:
    """Return the first adapter supporting ``task``."""
    for adapter in adapters:
        if adapter.supports(task):
            return adapter
    raise ValueError(f"No generate-and-evaluate adapter supports task {task!r}.")
