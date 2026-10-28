"""Prompt templates and registry bundled with JudgeArena."""

from judgearena.prompts.parsing import (
    JudgeParser,
    OutputParser,
    ParsedPreference,
    ParsedScore,
    register_parser,
)
from judgearena.prompts.registry import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    DEFAULT_WITH_EXPLANATION_PRESET,
    FASTCHAT_PAIRWISE_PROMPT_PRESET,
    FLUENCY_JUDGE_PROMPT_PRESET,
    JUDGE_PROMPT_PRESETS,
    MT_BENCH_101_CLEAN_PROMPT_PRESET,
    MT_BENCH_101_PROMPT_PRESET,
    PRESETS,
    JudgePromptPreset,
    ResolvedJudgePrompt,
    available_prompt_presets,
    default_preset_for_task,
    register_prompt,
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
    "ParsedScore",
    "OutputParser",
    "JudgeParser",
    "JudgePromptPreset",
    "ResolvedJudgePrompt",
    "available_prompt_presets",
    "register_prompt",
    "register_parser",
    "default_preset_for_task",
    "resolve_judge_prompt",
    "resolve_run_judge_prompt",
]
