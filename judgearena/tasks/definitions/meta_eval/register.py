"""Register this task family's reusable components."""

from pathlib import Path

from judgearena.prompts.parsing import register_parser, resolve_judge_parser
from judgearena.prompts.registry import (
    META_EVAL_ALPACA_EVAL_JSON_PROMPT_PRESET,
    META_EVAL_ALPACA_EVAL_PAIR_SCORE_PROMPT_PRESET,
    META_EVAL_PAIR_SCORE_PROMPT_PRESET,
    JudgePromptPreset,
    register_prompt,
)

HERE = Path(__file__).resolve().parent

register_parser(
    "meta-eval-score", resolve_judge_parser(f"{HERE}/parsers.py:PAIR_SCORE")
)
register_parser(
    "alpaca-eval-json", resolve_judge_parser(f"{HERE}/parsers.py:ALPACA_JSON")
)

register_prompt(
    JudgePromptPreset(
        name=META_EVAL_PAIR_SCORE_PROMPT_PRESET,
        parser=resolve_judge_parser("meta-eval-score"),
        system_file="system-prompt.txt",
        user_file="prompt.txt",
        output_format_file="integer-score-output.txt",
    )
)

register_prompt(
    JudgePromptPreset.from_file(
        META_EVAL_ALPACA_EVAL_JSON_PROMPT_PRESET,
        HERE / "prompts/alpaca-json.yaml",
    )
)

register_prompt(
    JudgePromptPreset.from_file(
        META_EVAL_ALPACA_EVAL_PAIR_SCORE_PROMPT_PRESET,
        HERE / "prompts/alpaca-score.yaml",
    )
)
