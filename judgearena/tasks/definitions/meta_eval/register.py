"""Register this task family's reusable components."""

from judgearena.prompts.parsing import PairScore, register_parser, resolve_judge_parser
from judgearena.prompts.registry import (
    JudgePromptPreset,
    register_prompt,
)

register_parser("meta-eval-score", PairScore(temperature=0.5))

register_prompt(
    JudgePromptPreset(
        name="meta-eval-pair-score",
        parser=resolve_judge_parser("meta-eval-score"),
        system_file="system-prompt.txt",
        user_file="prompt.txt",
        output_format_file="integer-score-output.txt",
    )
)
