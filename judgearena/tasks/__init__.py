"""Public API for declaring, validating, and discovering benchmark tasks."""

from judgearena.tasks.registry import TaskRegistry, get_packaged_task
from judgearena.tasks.schema import ResolvedTaskSpec, TaskSpec

__all__ = ["ResolvedTaskSpec", "TaskRegistry", "TaskSpec", "get_packaged_task"]
