from __future__ import annotations

import math
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

import pandas as pd

from judgearena.benchmarks.mt_bench.common import (
    is_reference_based_category,
    resolve_mt_bench_turn_flags,
)
from judgearena.benchmarks.mt_bench.pairwise_judging import (
    MTBenchJudgeItem,
    build_mt_bench_pairwise_judge_items,
    infer_pairwise_judgments_by_prompt_groups,
)
from judgearena.benchmarks.mt_bench.prompt_templates import (
    build_mt_bench_user_prompt_template,
)
from judgearena.prompts.parsing import JudgeParser, parser_name
from judgearena.prompts.registry import (
    ResolvedJudgePrompt,
)

MTBenchPromptResolver = Callable[[bool], ResolvedJudgePrompt]


@dataclass(frozen=True)
class MTBenchPresetPrompt:
    name: str
    preset_name: str
    parse: JudgeParser
    system_prompt: str | None
    user_prompt_template: str
    multi_turn: bool
    ref_based: bool


def _extract_judge_instructions(user_prompt_template: str) -> str:
    markers = ("# Your task", "# Your output")
    marker_indices = [
        index for marker in markers if (index := user_prompt_template.find(marker)) >= 0
    ]
    if not marker_indices:
        raise ValueError(
            "Could not find '# Your task' or '# Your output' in preset template."
        )
    return user_prompt_template[min(marker_indices) :].lstrip()


def _build_mt_bench_preset_user_prompt_template(
    *,
    resolved_prompt: ResolvedJudgePrompt,
    multi_turn: bool,
    ref_based: bool,
) -> str:
    base_template = build_mt_bench_user_prompt_template(
        multi_turn=multi_turn,
        ref_based=ref_based,
    )
    judge_instructions = _extract_judge_instructions(
        resolved_prompt.user_prompt_template
    )
    return f"{base_template}\n\n{judge_instructions}"


def _build_mt_bench_prompt(
    category: str | None,
    multi_turn: bool,
    *,
    reference_categories: Collection[str],
    resolved_prompt: ResolvedJudgePrompt,
) -> MTBenchPresetPrompt:
    ref_based = is_reference_based_category(category, reference_categories)
    if resolved_prompt.delegated:
        raise ValueError(
            f"Judge prompt preset '{resolved_prompt.preset_name}' is delegated and "
            "cannot be used for MT-Bench preset judging."
        )
    if resolved_prompt.parse is None:
        raise ValueError(
            f"Judge prompt preset '{resolved_prompt.preset_name}' has no parser."
        )
    suffix = "multi" if multi_turn else "single"
    if ref_based:
        suffix += "_ref"
    return MTBenchPresetPrompt(
        name=f"{resolved_prompt.preset_name}-{suffix}",
        preset_name=resolved_prompt.preset_name,
        parse=resolved_prompt.parser,
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=_build_mt_bench_preset_user_prompt_template(
            resolved_prompt=resolved_prompt,
            multi_turn=multi_turn,
            ref_based=ref_based,
        ),
        multi_turn=multi_turn,
        ref_based=ref_based,
    )


def _build_mt_bench_preset_items(
    *,
    questions: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    eval_single: bool,
    eval_multi: bool,
    truncate_input_chars: int | None,
    reference_categories: Collection[str],
    prompt_for_turn: MTBenchPromptResolver,
    strip_thinking_before_judging: bool = False,
) -> list[MTBenchJudgeItem]:
    single_turn_prompt = prompt_for_turn(False) if eval_single else None
    multi_turn_prompt = prompt_for_turn(True) if eval_multi else None

    def select_prompt(category: str | None, multi_turn: bool) -> MTBenchPresetPrompt:
        resolved_prompt = multi_turn_prompt if multi_turn else single_turn_prompt
        if resolved_prompt is None:
            turn_name = "multi-turn" if multi_turn else "single-turn"
            raise ValueError(f"Prompt requested for disabled {turn_name} judging.")
        return _build_mt_bench_prompt(
            category,
            multi_turn=multi_turn,
            reference_categories=reference_categories,
            resolved_prompt=resolved_prompt,
        )

    return build_mt_bench_pairwise_judge_items(
        questions=questions,
        completions_a=completions_a,
        completions_b=completions_b,
        eval_single=eval_single,
        eval_multi=eval_multi,
        truncate_input_chars=truncate_input_chars,
        select_prompt=select_prompt,
        strip_thinking_before_judging=strip_thinking_before_judging,
    )


def _normalize_preference(preference: float | None, *, swapped: bool) -> float:
    if preference is None:
        return math.nan
    return 1.0 - preference if swapped else float(preference)


def judge_mt_bench_with_preset(
    *,
    judge_chat_model,
    judge_model: str,
    questions: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    model_a: str,
    model_b: str,
    turns_mode: str,
    swap_mode: str,
    truncate_input_chars: int | None,
    use_tqdm: bool,
    reference_categories: Collection[str],
    prompt_for_turn: MTBenchPromptResolver,
    strip_thinking_before_judging: bool = False,
) -> tuple[pd.Series, list[dict[str, Any]], list[dict[str, object]]]:
    assert swap_mode in ("fixed", "both")
    eval_single, eval_multi = resolve_mt_bench_turn_flags(turns_mode)

    items = _build_mt_bench_preset_items(
        questions=questions,
        completions_a=completions_a,
        completions_b=completions_b,
        eval_single=eval_single,
        eval_multi=eval_multi,
        truncate_input_chars=truncate_input_chars,
        reference_categories=reference_categories,
        prompt_for_turn=prompt_for_turn,
        strip_thinking_before_judging=strip_thinking_before_judging,
    )
    judgments, prompt_kwargs_used = infer_pairwise_judgments_by_prompt_groups(
        judge_chat_model=judge_chat_model,
        items=items,
        use_tqdm=use_tqdm,
        swap_answers=False,
    )

    annotations: list[dict[str, Any]] = []
    metadata: list[dict[str, object]] = []
    preferences: list[float] = []

    def _append_results(
        raw_judgments: list[str],
        used_prompt_kwargs: list[dict[str, str]],
        *,
        swapped: bool,
    ) -> None:
        for item, raw_judgment, prompt_kwargs in zip(
            items, raw_judgments, used_prompt_kwargs, strict=True
        ):
            prompt: MTBenchPresetPrompt = item.prompt
            parsed_preference = prompt.parse(raw_judgment)
            normalized_preference = _normalize_preference(
                parsed_preference,
                swapped=swapped,
            )
            annotations.append(
                {
                    "question_id": item.question_id,
                    "category": item.category,
                    "turn": item.turn,
                    "model_A": model_b if swapped else model_a,
                    "model_B": model_a if swapped else model_b,
                    "judge": judge_model,
                    "prompt_name": prompt.name,
                    "prompt_preset": prompt.preset_name,
                    "parser_mode": parser_name(prompt.parse),
                    "system_prompt": prompt.system_prompt,
                    "user_prompt_template": prompt.user_prompt_template,
                    "user_prompt": prompt.user_prompt_template.format(**prompt_kwargs),
                    "judge_completion": raw_judgment,
                    "preference": normalized_preference,
                    "swapped": swapped,
                }
            )
            metadata.append(
                {
                    "question_id": item.question_id,
                    "category": item.category,
                    "turn": item.turn,
                }
            )
            preferences.append(normalized_preference)

    _append_results(judgments, prompt_kwargs_used, swapped=False)

    if swap_mode == "both":
        # swap_mode="both": append the inverted swapped-order scores as
        # additional data points (see _normalize_preference(swapped=True)).
        swapped_judgments, swapped_prompt_kwargs = (
            infer_pairwise_judgments_by_prompt_groups(
                judge_chat_model=judge_chat_model,
                items=items,
                use_tqdm=use_tqdm,
                swap_answers=True,
            )
        )
        _append_results(swapped_judgments, swapped_prompt_kwargs, swapped=True)

    return pd.Series(preferences, dtype=float), annotations, metadata
