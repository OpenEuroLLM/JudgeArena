"""Prompt templates bundled with JudgeArena.

The :mod:`judgearena.prompts.registry` submodule exposes the named presets
used by the judge plus a per-task default mapping; see ``--judge_prompt_preset``
on the CLI.
"""

from judgearena.prompts.registry import (
    PRESETS,
    TASK_DEFAULT_PRESET,
    ResolvedJudgePrompt,
    default_preset_for_task,
    resolve_judge_prompt,
)

__all__ = [
    "PRESETS",
    "TASK_DEFAULT_PRESET",
    "ResolvedJudgePrompt",
    "default_preset_for_task",
    "resolve_judge_prompt",
]
