"""Public API for declaring, validating, and discovering benchmark tasks."""

from judgearena.tasks.registry import get_packaged_task, load_tasks
from judgearena.tasks.schema import ResolvedTaskSpec, TaskSpec

__all__ = ["ResolvedTaskSpec", "TaskSpec", "get_packaged_task", "load_tasks"]
