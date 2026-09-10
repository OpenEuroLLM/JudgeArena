"""Helpers shared by runners that read human arena battles."""

from __future__ import annotations

from judgearena.tasks.schema import ResolvedTaskSpec


def resolve_task_languages(
    task: ResolvedTaskSpec, requested: list[str] | None, *, setting: str
) -> list[str]:
    """Narrow an optional runtime filter within a task language variant."""
    selected = list(requested or [])
    if task.selection is None:
        return selected

    variant_languages = list(task.selection.values)
    if not selected:
        return variant_languages
    narrowed = [language for language in selected if language in variant_languages]
    if not narrowed:
        raise ValueError(
            f"{setting} {requested} has no overlap with the languages of task "
            f"{task.task!r} ({variant_languages})."
        )
    return narrowed
