"""Prompt templates and registry bundled with JudgeArena."""

from judgearena.prompts.registry import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    DEFAULT_WITH_EXPLANATION_PRESET,
    FASTCHAT_PAIRWISE_PROMPT_PRESET,
    FLUENCY_JUDGE_PROMPT_PRESET,
    JUDGE_PROMPT_PRESETS,
    PRESETS,
    TASK_DEFAULT_PRESET,
    JudgeParserMode,
    JudgePromptPreset,
    ResolvedJudgePrompt,
    default_preset_for_task,
    resolve_judge_prompt,
    resolve_run_judge_prompt,
)

__all__ = [
    "DEFAULT_JUDGE_PROMPT_PRESET",
    "DEFAULT_WITH_EXPLANATION_PRESET",
    "FASTCHAT_PAIRWISE_PROMPT_PRESET",
    "FLUENCY_JUDGE_PROMPT_PRESET",
    "JUDGE_PROMPT_PRESETS",
    "PRESETS",
    "TASK_DEFAULT_PRESET",
    "JudgeParserMode",
    "JudgePromptPreset",
    "ResolvedJudgePrompt",
    "default_preset_for_task",
    "resolve_judge_prompt",
    "resolve_run_judge_prompt",
]
