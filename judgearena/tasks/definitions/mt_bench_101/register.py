"""Register this task family's reusable prompts and parsers."""

from judgearena.prompts.registry import (
    MT_BENCH_101_CLEAN_PROMPT_PRESET,
    MT_BENCH_101_PROMPT_PRESET,
    JudgePromptPreset,
    register_prompt,
)


def _render_mt_bench_101_scores(samples, **kwargs):
    from judgearena.prompts.mt_bench_101 import render_mt_bench_101_requests

    return render_mt_bench_101_requests(samples, **kwargs)


register_prompt(
    JudgePromptPreset(
        name=MT_BENCH_101_PROMPT_PRESET,
        judging_mode="single_answer",
        delegated=True,
        score_renderer=_render_mt_bench_101_scores,
    )
)

register_prompt(
    JudgePromptPreset(
        name=MT_BENCH_101_CLEAN_PROMPT_PRESET,
        judging_mode="single_answer",
        delegated=True,
        score_renderer=_render_mt_bench_101_scores,
    )
)
