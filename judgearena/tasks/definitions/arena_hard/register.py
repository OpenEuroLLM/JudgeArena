"""Register this task family's reusable components."""

from pathlib import Path

from judgearena.benchmarks.scoring import register_metric
from judgearena.datasets.registry import register_dataset
from judgearena.prompts.parsing import register_parser, resolve_judge_parser
from judgearena.prompts.registry import (
    ARENA_HARD_CREATIVE_JUDGE_PROMPT_PRESET,
    ARENA_HARD_JUDGE_PROMPT_PRESET,
    JudgePromptPreset,
    register_prompt,
)
from judgearena.tasks.definitions.arena_hard.dataset import ArenaHardAdapter
from judgearena.tasks.definitions.arena_hard.metrics import (
    ArenaHardV01Metric,
    ArenaHardV20Metric,
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


register_dataset("arena_hard")(ArenaHardAdapter)


register_metric("arena_hard_v01")(ArenaHardV01Metric)
register_metric("arena_hard_v20")(ArenaHardV20Metric)
