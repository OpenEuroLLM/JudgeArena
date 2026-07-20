"""Public API for declaring, validating, and discovering benchmark tasks."""

from judgearena.tasks.registry import get_packaged_task, load_tasks, resolve_task
from judgearena.tasks.schema import ResolvedTaskSpec, TaskSelection, TaskSpec

__all__ = [
    "ResolvedTaskSpec",
    "TaskSelection",
    "TaskSpec",
    "get_packaged_task",
    "load_tasks",
    "resolve_task",
]
