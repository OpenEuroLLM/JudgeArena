"""Register this task family's reusable components."""

from judgearena.prompts.parsing import PairScore, register_parser, resolve_judge_parser
from judgearena.prompts.registry import (
    META_EVAL_PAIR_SCORE_PROMPT_PRESET,
    JudgePromptPreset,
    register_prompt,
)

register_parser("meta-eval-score", PairScore(temperature=0.5))

register_prompt(
    JudgePromptPreset(
        name=META_EVAL_PAIR_SCORE_PROMPT_PRESET,
        parser=resolve_judge_parser("meta-eval-score"),
        system_file="system-prompt.txt",
        user_file="prompt.txt",
        output_format_file="integer-score-output.txt",
    )
)
