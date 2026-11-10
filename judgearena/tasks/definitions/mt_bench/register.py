"""Register this task family's reusable components."""

from pathlib import Path

from judgearena.datasets.registry import register_dataset
from judgearena.prompts.parsing import register_parser, resolve_judge_parser
from judgearena.prompts.registry import (
    JudgePromptPreset,
    register_prompt,
)
from judgearena.tasks.definitions.mt_bench.dataset import MTBenchAdapter

HERE = Path(__file__).resolve().parent

register_parser("fastchat-verdict", resolve_judge_parser(f"{HERE}/parsers.py:VERDICT"))


def _prepare_mt_bench_prompts(cfg, prompt, instructions):
    from judgearena.tasks.definitions.mt_bench.rendering import prepare_mt_bench_prompts

    return prepare_mt_bench_prompts(cfg, prompt, instructions)


register_prompt(
    JudgePromptPreset(
        name="fastchat-pairwise",
        delegated=True,
        prepare_prompts=_prepare_mt_bench_prompts,
        aggregation="agreement",
    )
)


register_dataset("mt_bench")(MTBenchAdapter)
