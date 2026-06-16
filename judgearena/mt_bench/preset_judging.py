from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from judgearena.evaluate import PairScore
from judgearena.mt_bench.common import (
    is_reference_based_category,
    resolve_mt_bench_turn_flags,
)
from judgearena.mt_bench.pairwise_judging import (
    MTBenchJudgeItem,
    build_mt_bench_pairwise_judge_items,
    infer_pairwise_judgments_by_prompt_groups,
)
from judgearena.mt_bench.prompt_templates import build_mt_bench_user_prompt_template
from judgearena.prompts.registry import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    ResolvedJudgePrompt,
    resolve_judge_prompt,
)


@dataclass(frozen=True)
class MTBenchPresetPrompt:
    name: str
    preset_name: str
    parser_mode: str
    system_prompt: str | None
    user_prompt_template: str
    multi_turn: bool
    ref_based: bool


def _extract_output_section(user_prompt_template: str) -> str:
    marker = "# Your output"
    marker_index = user_prompt_template.find(marker)
    if marker_index < 0:
        raise ValueError("Could not find '# Your output' section in preset template.")
    return user_prompt_template[marker_index:].lstrip()


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
    output_section = _extract_output_section(resolved_prompt.user_prompt_template)
    return f"{base_template}\n\n{output_section}"


def _select_preset_prompt(
    category: str | None,
    multi_turn: bool,
    *,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    provide_explanation: bool,
    system_file: str | None = None,
    user_file: str | None = None,
) -> MTBenchPresetPrompt:
    ref_based = is_reference_based_category(category)
    resolved_prompt = resolve_judge_prompt(
        preset=prompt_preset,
        system_file=system_file,
        user_file=user_file,
        provide_explanation=provide_explanation,
        multi_turn=multi_turn,
    )
    if resolved_prompt.delegated:
        raise ValueError(
            f"Judge prompt preset '{resolved_prompt.preset_name}' is delegated and "
            "cannot be used for MT-Bench preset judging."
        )
    suffix = "multi" if multi_turn else "single"
    if ref_based:
        suffix += "_ref"
    return MTBenchPresetPrompt(
        name=f"{resolved_prompt.preset_name}-{suffix}",
        preset_name=resolved_prompt.preset_name,
        parser_mode=resolved_prompt.parser_mode,
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
    prompt_preset: str,
    provide_explanation: bool,
    system_file: str | None = None,
    user_file: str | None = None,
) -> list[MTBenchJudgeItem]:
    return build_mt_bench_pairwise_judge_items(
        questions=questions,
        completions_a=completions_a,
        completions_b=completions_b,
        eval_single=eval_single,
        eval_multi=eval_multi,
        truncate_input_chars=truncate_input_chars,
        select_prompt=lambda category, multi_turn: _select_preset_prompt(
            category,
            multi_turn=multi_turn,
            prompt_preset=prompt_preset,
            provide_explanation=provide_explanation,
            system_file=system_file,
            user_file=user_file,
        ),
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
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    provide_explanation: bool = False,
    system_file: str | None = None,
    user_file: str | None = None,
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
        prompt_preset=prompt_preset,
        provide_explanation=provide_explanation,
        system_file=system_file,
        user_file=user_file,
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
            parsed_preference = PairScore(
                parser_mode=prompt.parser_mode
            ).parse_model_raw(raw_judgment)
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
                    "parser_mode": prompt.parser_mode,
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
