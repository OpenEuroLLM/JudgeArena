"""Register this task family's reusable prompts and parsers."""

from pathlib import Path

from judgearena.prompts.parsing import register_parser, resolve_judge_parser
from judgearena.prompts.registry import (
    ARENA_HARD_CREATIVE_JUDGE_PROMPT_PRESET,
    ARENA_HARD_JUDGE_PROMPT_PRESET,
    JudgePromptPreset,
    register_prompt,
)

HERE = Path(__file__).resolve().parent

register_parser(
    "arena-hard-verdict", resolve_judge_parser(f"{HERE}/parsers.py:ARENA_HARD_VERDICT")
)

register_prompt(
    JudgePromptPreset.from_file(
        ARENA_HARD_JUDGE_PROMPT_PRESET,
        HERE / "prompts/pairwise.yaml",
    )
)

register_prompt(
    JudgePromptPreset.from_file(
        ARENA_HARD_CREATIVE_JUDGE_PROMPT_PRESET,
        HERE / "prompts/creative.yaml",
    )
)
