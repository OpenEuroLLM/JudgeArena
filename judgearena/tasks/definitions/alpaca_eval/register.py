"""Register this task family's reusable components."""

from pathlib import Path

from judgearena.benchmarks.scoring import register_metric
from judgearena.prompts.parsing import register_parser, resolve_judge_parser
from judgearena.prompts.registry import (
    JudgePromptPreset,
    register_prompt,
)
from judgearena.tasks.definitions.alpaca_eval.metrics import (
    AlpacaEvalLengthControlledMetric,
)

HERE = Path(__file__).resolve().parent

register_parser(
    "alpaca-eval-token", resolve_judge_parser(f"{HERE}/parsers.py:ALPACA_EVAL_TOKEN")
)

register_prompt(
    JudgePromptPreset.from_file("alpaca-eval", HERE / "prompts/pairwise.yaml")
)


register_metric("alpaca_eval_length_controlled")(AlpacaEvalLengthControlledMetric)

register_parser(
    "alpaca-eval-json", resolve_judge_parser(f"{HERE}/parsers.py:ALPACA_JSON")
)
register_prompt(
    JudgePromptPreset.from_file("alpaca-eval-json", HERE / "prompts/json.yaml")
)
register_prompt(
    JudgePromptPreset.from_file("alpaca-eval-pair-score", HERE / "prompts/score.yaml")
)
