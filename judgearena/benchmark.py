"""Shared contract and model setup for generate-and-evaluate benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from judgearena.models import (
    build_default_judge_model_kwargs,
    is_thinking_model,
    make_model,
)

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


def build_generation_kwargs(
    cfg: RunConfig, model_spec: str, *, role: Literal["A", "B"]
) -> dict[str, object]:
    """Resolve generation kwargs for an evaluated or baseline model."""
    if role == "A":
        generation_kwargs = cfg.model.evaluated_generation_kwargs()
    elif role == "B":
        generation_kwargs = cfg.model.baseline_generation_kwargs()
    else:  # guarded by the type; keeps untyped callers honest
        raise ValueError(f"Unknown generation role: {role!r}")

    provider, _, model_name = model_spec.partition("/")
    budget = cfg.judge.battle_thinking_token_budget
    if budget is not None and provider == "VLLM" and is_thinking_model(model_name):
        max_tokens = int(generation_kwargs.get("max_tokens", cfg.model.max_out_tokens))
        generation_kwargs["thinking_token_budget"] = min(int(budget), max_tokens)
    return generation_kwargs


def build_judge(cfg: RunConfig):
    """Construct the configured judge consistently across benchmark adapters."""
    return make_model(
        model=cfg.judge.model,
        **build_default_judge_model_kwargs(
            cfg.judge.model,
            cfg.model.engine_kwargs,
            judge_engine_kwargs_override=cfg.judge.model_kwargs(
                fallback_chat_template=cfg.model.chat_template,
            ),
        ),
    )
