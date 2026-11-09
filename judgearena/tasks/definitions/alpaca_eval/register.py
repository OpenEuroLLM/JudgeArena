"""Register this task family's reusable prompts and parsers."""

from pathlib import Path

from judgearena.prompts.parsing import register_parser, resolve_judge_parser
from judgearena.prompts.registry import (
    ALPACA_EVAL_JUDGE_PROMPT_PRESET,
    JudgePromptPreset,
    register_prompt,
)

HERE = Path(__file__).resolve().parent

register_parser(
    "alpaca-eval-token", resolve_judge_parser(f"{HERE}/parsers.py:ALPACA_EVAL_TOKEN")
)

register_prompt(
    JudgePromptPreset.from_file(
        ALPACA_EVAL_JUDGE_PROMPT_PRESET,
        HERE / "prompts/pairwise.yaml",
    )
)
