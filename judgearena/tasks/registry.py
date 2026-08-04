"""Discover task definitions and validate their referenced component IDs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable

from judgearena.prompts.registry import JUDGE_PROMPT_PRESETS
from judgearena.tasks.loader import TaskDefinitionError, TaskLoader
from judgearena.tasks.schema import ResolvedTaskSpec


@dataclass(frozen=True)
class AdapterCatalog:
    """Component IDs that task YAML files may reference."""

    runners: frozenset[str] = frozenset({"pairwise"})
    datasets: frozenset[str] = frozenset({"judgearena_tables"})
    prompts: frozenset[str] = frozenset(JUDGE_PROMPT_PRESETS)
    parsers: frozenset[str] = frozenset({"pairwise_preference"})
    scorers: frozenset[str] = frozenset({"pairwise_win_rate"})


def load_tasks(
    definitions_root: Traversable | None = None,
    *,
    adapters: AdapterCatalog | None = None,
) -> dict[str, ResolvedTaskSpec]:
    """Discover, validate, and return all packaged tasks keyed by task ID.

    With no arguments this reads JudgeArena's installed definitions and caches
    the result. An explicit ``definitions_root`` (used by tests) is never cached.
    """
    if definitions_root is None and adapters is None:
        return _load_packaged_tasks()
    root = definitions_root or files("judgearena.tasks").joinpath("definitions")
    return _discover_tasks(root, adapters or AdapterCatalog())


@cache
def _load_packaged_tasks() -> dict[str, ResolvedTaskSpec]:
    root = files("judgearena.tasks").joinpath("definitions")
    return _discover_tasks(root, AdapterCatalog())


def _discover_tasks(
    definitions_root: Traversable, adapters: AdapterCatalog
) -> dict[str, ResolvedTaskSpec]:
    loader = TaskLoader(definitions_root)
    tasks: dict[str, ResolvedTaskSpec] = {}
    for relative_path in loader.discover():
        resolved = loader.load(relative_path)
        if resolved.task in tasks:
            other = tasks[resolved.task].provenance.source_path
            raise TaskDefinitionError(
                f"Duplicate task ID {resolved.task!r} in {other} and {relative_path}"
            )
        _validate_adapter_ids(resolved, adapters)
        tasks[resolved.task] = resolved
    return dict(sorted(tasks.items()))


def _validate_adapter_ids(resolved: ResolvedTaskSpec, adapters: AdapterCatalog) -> None:
    spec = resolved.spec
    references = {
        "runner": (spec.protocol.runner, adapters.runners),
        "dataset adapter": (spec.dataset.adapter, adapters.datasets),
        "prompt": (spec.protocol.judge.default_prompt, adapters.prompts),
        "parser": (spec.protocol.judge.parser, adapters.parsers),
        "scorer": (spec.protocol.scoring.adapter, adapters.scorers),
    }
    for kind, (adapter_id, available) in references.items():
        if adapter_id not in available:
            raise TaskDefinitionError(
                f"{resolved.provenance.source_path}: unknown {kind} {adapter_id!r}"
            )


def get_packaged_task(task_id: str) -> ResolvedTaskSpec | None:
    """Look up a task from JudgeArena's installed YAML definitions."""
    return load_tasks().get(task_id)
