"""Helpers shared by the runners that read human arena battles."""

from __future__ import annotations

from judgearena.tasks.schema import ResolvedTaskSpec


def resolve_task_languages(
    task: ResolvedTaskSpec, requested: list[str] | None, *, setting: str
) -> list[str]:
    """Narrow a task variant's languages by an explicit runtime filter.

    A variant such as ``elo-lmarena-140k-en`` already preselects languages, so
    the runtime filter may only narrow within that selection. ``setting`` names
    the config field in the error message.
    """
    selected = list(requested or [])
    if task.selection is None:
        return selected

    variant_languages = list(task.selection.values)
    if not selected:
        return variant_languages
    narrowed = [lang for lang in selected if lang in set(variant_languages)]
    if not narrowed:
        raise ValueError(
            f"{setting} {requested} has no overlap with the languages of task "
            f"{task.task!r} ({variant_languages})."
        )
    return narrowed
