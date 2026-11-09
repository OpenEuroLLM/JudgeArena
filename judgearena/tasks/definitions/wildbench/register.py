"""Register this task family's reusable components."""

from pathlib import Path

from judgearena.benchmarks.scoring import register_metric
from judgearena.datasets.registry import register_dataset
from judgearena.prompts.parsing import register_parser, resolve_judge_parser
from judgearena.prompts.registry import JudgePromptPreset, register_prompt
from judgearena.tasks.definitions.wildbench.dataset import WildBenchAdapter
from judgearena.tasks.definitions.wildbench.metrics import (
    WildBenchRewardMetric,
    WildBenchRewardMixMetric,
    WildBenchScoreMetric,
)

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


register_dataset("wildbench")(WildBenchAdapter)


register_metric("wildbench_reward")(WildBenchRewardMetric)
register_metric("wildbench_reward_mix")(WildBenchRewardMixMetric)
register_metric("wildbench_score")(WildBenchScoreMetric)
