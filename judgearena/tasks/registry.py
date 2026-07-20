"""Discover task definitions and validate their referenced component IDs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable

from judgearena.prompts.registry import JUDGE_PROMPT_PRESETS
from judgearena.tasks.loader import TaskDefinitionError, TaskLoader
from judgearena.tasks.schema import ResolvedTaskSpec, TaskSelection


@dataclass(frozen=True)
class AdapterCatalog:
    """Component IDs that task YAML files may reference."""

    runners: frozenset[str] = frozenset({"pairwise"})
    datasets: frozenset[str] = frozenset(
        {"arena_hard", "judgearena_tables", "m_arena_hard"}
    )
    prompts: frozenset[str] = frozenset(JUDGE_PROMPT_PRESETS)
    parsers: frozenset[str] = frozenset({"pairwise_preference"})
    scorers: frozenset[str] = frozenset({"pairwise_win_rate"})


@dataclass(frozen=True)
class TaskSummary:
    """Small task description used by ``judgearena tasks list``."""

    task: str
    task_version: int
    description: str
    tags: tuple[str, ...]
    source_path: str


@dataclass(frozen=True)
class ValidationReport:
    """Summary returned after validating every packaged task."""

    tasks: tuple[TaskSummary, ...]

    @property
    def count(self) -> int:
        return len(self.tasks)


class UnknownTaskError(KeyError):
    """Raised when a task ID is absent from the packaged registry."""

    pass


class TaskRegistry:
    """Discover, validate, and look up packaged task definitions by ID."""

    def __init__(
        self,
        definitions_root: Traversable | None = None,
        *,
        adapters: AdapterCatalog | None = None,
    ):
        if definitions_root is None:
            definitions_root = files("judgearena.tasks").joinpath("definitions")
        self._loader = TaskLoader(definitions_root)
        self._adapters = adapters or AdapterCatalog()
        self._tasks: dict[str, ResolvedTaskSpec] | None = None

    def get(self, task_id: str) -> ResolvedTaskSpec:
        """Return one validated task definition."""
        resolved = self.find(task_id)
        if resolved is not None:
            return resolved
        known = ", ".join(sorted(self._load_all())) or "none"
        raise UnknownTaskError(f"Unknown task {task_id!r}; registered tasks: {known}")

    def find(self, task_id: str) -> ResolvedTaskSpec | None:
        """Return a task definition, or ``None`` when it is not registered."""
        tasks = self._load_all()
        exact = tasks.get(task_id)
        if exact is not None:
            return exact

        for task in sorted(
            tasks.values(), key=lambda item: len(item.task), reverse=True
        ):
            variants = task.spec.variants
            if variants is None:
                continue
            prefix = f"{task.task}{variants.separator}"
            if not task_id.startswith(prefix):
                continue
            suffix = task_id[len(prefix) :]
            if suffix in variants.values:
                values = (suffix,)
            elif suffix in variants.groups:
                values = variants.groups[suffix]
            else:
                continue
            return ResolvedTaskSpec(
                spec=task.spec,
                provenance=task.provenance,
                invocation_task=task_id,
                selection=TaskSelection(
                    selector=variants.selector,
                    name=suffix,
                    values=values,
                ),
            )
        return None

    def list(self) -> tuple[TaskSummary, ...]:
        """Return concise summaries of all packaged tasks."""
        return tuple(self._summary(task) for task in self._load_all().values())

    def validate_all(self) -> ValidationReport:
        """Validate all packaged tasks and return their summaries."""
        return ValidationReport(self.list())

    def _load_all(self) -> dict[str, ResolvedTaskSpec]:
        if self._tasks is not None:
            return self._tasks
        tasks: dict[str, ResolvedTaskSpec] = {}
        for relative_path in self._loader.discover():
            resolved = self._loader.load(relative_path)
            if resolved.task in tasks:
                other = tasks[resolved.task].provenance.source_path
                raise TaskDefinitionError(
                    f"Duplicate task ID {resolved.task!r} in {other} and "
                    f"{relative_path}"
                )
            self._validate_adapter_ids(resolved)
            tasks[resolved.task] = resolved
        self._validate_variant_ids(tasks)
        self._tasks = dict(sorted(tasks.items()))
        return self._tasks

    @staticmethod
    def _validate_variant_ids(tasks: dict[str, ResolvedTaskSpec]) -> None:
        aliases: dict[str, str] = {}
        for task_id, resolved in tasks.items():
            variants = resolved.spec.variants
            if variants is None:
                continue
            for suffix in (*variants.values, *variants.groups):
                alias = f"{task_id}{variants.separator}{suffix}"
                owner = aliases.get(alias)
                if alias in tasks or owner is not None:
                    other = alias if alias in tasks else owner
                    raise TaskDefinitionError(
                        f"Variant task ID {alias!r} from {task_id!r} conflicts "
                        f"with {other!r}."
                    )
                aliases[alias] = task_id

    def _validate_adapter_ids(self, resolved: ResolvedTaskSpec) -> None:
        spec = resolved.spec
        references = {
            "runner": (spec.protocol.runner, self._adapters.runners),
            "dataset adapter": (spec.dataset.adapter, self._adapters.datasets),
            "prompt": (spec.protocol.judge.default_prompt, self._adapters.prompts),
            "parser": (spec.protocol.judge.parser, self._adapters.parsers),
            "scorer": (spec.protocol.scoring.adapter, self._adapters.scorers),
        }
        for kind, (adapter_id, available) in references.items():
            if adapter_id not in available:
                raise TaskDefinitionError(
                    f"{resolved.provenance.source_path}: unknown {kind} {adapter_id!r}"
                )

    @staticmethod
    def _summary(resolved: ResolvedTaskSpec) -> TaskSummary:
        spec = resolved.spec
        return TaskSummary(
            task=spec.task,
            task_version=spec.task_version,
            description=spec.description,
            tags=spec.tags,
            source_path=resolved.provenance.source_path,
        )


_PACKAGED_TASKS = TaskRegistry()


def get_packaged_task(task_id: str) -> ResolvedTaskSpec | None:
    """Look up a task from JudgeArena's installed YAML definitions."""
    return _PACKAGED_TASKS.find(task_id)
