"""Register this task family's reusable prompts and parsers."""

from pathlib import Path

from judgearena.prompts.parsing import register_parser, resolve_judge_parser
from judgearena.prompts.registry import (
    FASTCHAT_PAIRWISE_PROMPT_PRESET,
    JudgePromptPreset,
    register_prompt,
)

HERE = Path(__file__).resolve().parent

register_parser("fastchat-verdict", resolve_judge_parser(f"{HERE}/parsers.py:VERDICT"))


def _prepare_mt_bench_prompts(cfg, prompt, instructions):
    from judgearena.prompts.mt_bench import prepare_mt_bench_prompts

    return prepare_mt_bench_prompts(cfg, prompt, instructions)


register_prompt(
    JudgePromptPreset(
        name=FASTCHAT_PAIRWISE_PROMPT_PRESET,
        delegated=True,
        prepare_prompts=_prepare_mt_bench_prompts,
        aggregation="agreement",
    )
)
