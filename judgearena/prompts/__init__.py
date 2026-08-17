"""Prompt templates and registry bundled with JudgeArena."""

from judgearena.prompts.registry import (
    CRITERIA_JUDGE_PROMPT_PRESET,
    DEFAULT_JUDGE_PROMPT_PRESET,
    DEFAULT_WITH_EXPLANATION_PRESET,
    FASTCHAT_PAIRWISE_PROMPT_PRESET,
    FLUENCY_JUDGE_PROMPT_PRESET,
    JUDGE_PROMPT_PRESETS,
    PRESETS,
    JudgeParser,
    JudgePromptPreset,
    ResolvedJudgePrompt,
    default_preset_for_task,
    resolve_judge_prompt,
    resolve_run_judge_prompt,
)

__all__ = [
    "CRITERIA_JUDGE_PROMPT_PRESET",
    "DEFAULT_JUDGE_PROMPT_PRESET",
    "DEFAULT_WITH_EXPLANATION_PRESET",
    "FASTCHAT_PAIRWISE_PROMPT_PRESET",
    "FLUENCY_JUDGE_PROMPT_PRESET",
    "JUDGE_PROMPT_PRESETS",
    "PRESETS",
    "JudgeParser",
    "JudgePromptPreset",
    "ResolvedJudgePrompt",
    "default_preset_for_task",
    "resolve_judge_prompt",
    "resolve_run_judge_prompt",
]
