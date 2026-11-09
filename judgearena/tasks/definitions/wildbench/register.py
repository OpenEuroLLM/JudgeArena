"""Register this task family's reusable prompts and parsers."""

from pathlib import Path

from judgearena.prompts.parsing import register_parser, resolve_judge_parser
from judgearena.prompts.registry import JudgePromptPreset, register_prompt

HERE = Path(__file__).resolve().parent

register_parser("wildbench-reward", resolve_judge_parser(f"{HERE}/parsers.py:REWARD"))
register_parser("wildbench-score", resolve_judge_parser(f"{HERE}/parsers.py:SCORE"))

register_prompt(
    JudgePromptPreset.from_file(
        "wildbench-score",
        HERE / "prompts/score.yaml",
    )
)

register_prompt(
    JudgePromptPreset.from_file(
        "wildbench-reward",
        HERE / "prompts/reward.yaml",
    )
)
