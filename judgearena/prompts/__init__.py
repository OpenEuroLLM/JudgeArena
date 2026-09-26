"""Prompt templates and registry bundled with JudgeArena."""

from judgearena.prompts.parsing import ParsedPreference
from judgearena.prompts.registry import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    DEFAULT_WITH_EXPLANATION_PRESET,
    FASTCHAT_PAIRWISE_PROMPT_PRESET,
    FLUENCY_JUDGE_PROMPT_PRESET,
    JUDGE_PROMPT_PRESETS,
    MT_BENCH_101_CLEAN_PROMPT_PRESET,
    MT_BENCH_101_PROMPT_PRESET,
    PRESETS,
    JudgeParser,
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
    "MT_BENCH_101_CLEAN_PROMPT_PRESET",
    "MT_BENCH_101_PROMPT_PRESET",
    "PRESETS",
    "ParsedPreference",
    "JudgeParser",
    "JudgePromptPreset",
    "ResolvedJudgePrompt",
    "default_preset_for_task",
    "resolve_judge_prompt",
    "resolve_run_judge_prompt",
]
