from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from judgearena.evaluate import (
    _PREFLIGHT_MAX_ITERATIONS,
    _PREFLIGHT_MIN_COMPLETION_CHARS,
    _PREFLIGHT_RESERVED_TOKENS,
    PairScore,
    _chars_per_token,
    _count_chat_tokens,
    _find_token_overflows,
)
from judgearena.judge_prompt_presets import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    ResolvedJudgePrompt,
    resolve_pairwise_judge_prompt,
)
from judgearena.log import get_logger
from judgearena.mt_bench.common import (
    MT_BENCH_REFERENCE_CATEGORIES,
    iter_mt_bench_pairwise_rows,
)
from judgearena.mt_bench.prompt_templates import build_mt_bench_user_prompt_template
from judgearena.openrouter_reference_pricing import OpenRouterReferencePricingTracker
from judgearena.utils import (
    LimitEventTracker,
    do_inference,
    strip_thinking_tags_with_metadata,
    truncate_with_metadata,
)

logger = get_logger(__name__)


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


def _extract_user_preamble(user_prompt_template: str) -> str:
    marker = "[User Question]"
    marker_index = user_prompt_template.find(marker)
    if marker_index < 0:
        raise ValueError("Could not find '[User Question]' section in preset template.")
    return user_prompt_template[:marker_index].rstrip()


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
    if resolved_prompt.system_prompt is None:
        user_preamble = _extract_user_preamble(resolved_prompt.user_prompt_template)
        return f"{user_preamble}\n\n{base_template}"
    output_section = _extract_output_section(resolved_prompt.user_prompt_template)
    return f"{base_template}\n\n{output_section}"


def _select_preset_prompt(
    category: str | None,
    multi_turn: bool,
    *,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    provide_explanation: bool,
) -> MTBenchPresetPrompt:
    ref_based = (category or "") in MT_BENCH_REFERENCE_CATEGORIES
    resolved_prompt = resolve_pairwise_judge_prompt(
        prompt_preset=prompt_preset,
        provide_explanation=provide_explanation,
        multi_turn=multi_turn,
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


def _group_indices_by_prompt(items: list[dict[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        grouped.setdefault(item["prompt_name"], []).append(idx)
    return grouped


def _swap_prompt_kwargs(kwargs: dict[str, str], *, multi_turn: bool) -> dict[str, str]:
    swapped = dict(kwargs)
    if multi_turn:
        swapped["answer_a_1"], swapped["answer_b_1"] = (
            swapped["answer_b_1"],
            swapped["answer_a_1"],
        )
        swapped["answer_a_2"], swapped["answer_b_2"] = (
            swapped["answer_b_2"],
            swapped["answer_a_2"],
        )
        return swapped
    swapped["answer_a"], swapped["answer_b"] = swapped["answer_b"], swapped["answer_a"]
    return swapped


def _build_chat_prompt_template(prompt: MTBenchPresetPrompt) -> ChatPromptTemplate:
    message_templates: list[tuple[str, str]] = []
    if prompt.system_prompt is not None:
        message_templates.append(("system", prompt.system_prompt))
    message_templates.append(("user", prompt.user_prompt_template))
    return ChatPromptTemplate.from_messages(message_templates)


def _answer_field_names(prompt: MTBenchPresetPrompt) -> tuple[str, ...]:
    if prompt.multi_turn:
        return ("answer_a_1", "answer_a_2", "answer_b_1", "answer_b_2")
    return ("answer_a", "answer_b")


def _truncation_flag_name(field: str) -> str:
    if field == "answer_a":
        return "answer_a_1_truncated"
    if field == "answer_b":
        return "answer_b_1_truncated"
    return f"{field}_truncated"


def _preflight_prompt_group_to_judge_budget(
    *,
    prompt_template: ChatPromptTemplate,
    prompt_kwargs_batch: list[dict[str, str]],
    batch_items: list[dict[str, Any]],
    judge_tokenizer: Any,
    max_judge_model_len: int,
    max_out_tokens_judge: int | None,
    limit_event_tracker: LimitEventTracker | None,
) -> list[Any]:
    prompt_inputs = prompt_template.batch(prompt_kwargs_batch)
    safe_budget = (
        max_judge_model_len - (max_out_tokens_judge or 0) - _PREFLIGHT_RESERVED_TOKENS
    )

    for _ in range(_PREFLIGHT_MAX_ITERATIONS):
        overflows = _find_token_overflows(prompt_inputs, judge_tokenizer, safe_budget)
        if not overflows:
            return prompt_inputs

        for idx, _token_count in overflows:
            prompt_kwargs = prompt_kwargs_batch[idx]
            item = batch_items[idx]
            answer_fields = item["answer_fields"]
            if not answer_fields:
                continue

            empty_kwargs = dict(prompt_kwargs)
            for field in answer_fields:
                empty_kwargs[field] = ""
            fixed_tokens = _count_chat_tokens(
                prompt_template.invoke(empty_kwargs),
                judge_tokenizer,
            )
            per_answer_budget = max(
                256, (safe_budget - fixed_tokens) // len(answer_fields)
            )

            for field in answer_fields:
                prompt_kwargs[field], shrunk = truncate_with_metadata(
                    prompt_kwargs[field],
                    max_len=max(
                        _PREFLIGHT_MIN_COMPLETION_CHARS,
                        int(
                            per_answer_budget
                            * _chars_per_token(prompt_kwargs[field], judge_tokenizer)
                            * 0.9
                        ),
                    ),
                    tracker=limit_event_tracker,
                    kind="judge_input_token_truncation",
                    stage="judge_input",
                    field=field,
                    case_id=item["case_id"],
                )
                if shrunk:
                    item["limit_flags"][_truncation_flag_name(field)] = True

        prompt_inputs = prompt_template.batch(prompt_kwargs_batch)

    final_overflows = _find_token_overflows(prompt_inputs, judge_tokenizer, safe_budget)
    for idx, token_count in final_overflows:
        if limit_event_tracker is not None:
            limit_event_tracker.record(
                "judge_input_token_truncation_failed",
                stage="judge_input",
                case_id=batch_items[idx]["case_id"],
                original_length=token_count,
                final_length=safe_budget,
                note=(
                    f"{_PREFLIGHT_MAX_ITERATIONS} shrink iterations did not "
                    f"bring tokens under {safe_budget}; falling through to "
                    "vLLM validation."
                ),
            )
    return prompt_inputs


def _infer_by_prompt_groups(
    *,
    judge_chat_model,
    items: list[dict[str, Any]],
    use_tqdm: bool,
    swap_answers: bool,
    judge_tokenizer: Any | None = None,
    max_judge_model_len: int | None = None,
    max_out_tokens_judge: int | None = None,
    usage_tracker: OpenRouterReferencePricingTracker | None = None,
    usage_phase: str | None = None,
    usage_model_spec: str | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    judgments: list[str] = [""] * len(items)
    used_prompt_kwargs: list[dict[str, str]] = [{} for _ in items]
    for idxs in _group_indices_by_prompt(items).values():
        prompt: MTBenchPresetPrompt = items[idxs[0]]["prompt"]
        prompt_template = _build_chat_prompt_template(prompt)

        batch_kwargs: list[dict[str, str]] = []
        batch_items = [items[item_index] for item_index in idxs]
        for item_index in idxs:
            prompt_kwargs = dict(items[item_index]["prompt_kwargs"])
            if swap_answers:
                prompt_kwargs = _swap_prompt_kwargs(
                    prompt_kwargs,
                    multi_turn=prompt.multi_turn,
                )
            batch_kwargs.append(prompt_kwargs)

        if judge_tokenizer is not None and max_judge_model_len is not None:
            prompt_inputs = _preflight_prompt_group_to_judge_budget(
                prompt_template=prompt_template,
                prompt_kwargs_batch=batch_kwargs,
                batch_items=batch_items,
                judge_tokenizer=judge_tokenizer,
                max_judge_model_len=max_judge_model_len,
                max_out_tokens_judge=max_out_tokens_judge,
                limit_event_tracker=items[idxs[0]].get("limit_event_tracker"),
            )
        else:
            prompt_inputs = prompt_template.batch(batch_kwargs)
        outputs = do_inference(
            chat_model=judge_chat_model,
            inputs=prompt_inputs,
            use_tqdm=use_tqdm,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
            usage_model_spec=usage_model_spec,
        )
        for item_index, output, prompt_kwargs in zip(
            idxs, outputs, batch_kwargs, strict=True
        ):
            judgments[item_index] = str(output)
            used_prompt_kwargs[item_index] = prompt_kwargs
    return judgments, used_prompt_kwargs


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
    strip_thinking_before_judging: bool,
    limit_event_tracker: LimitEventTracker | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    truncated_field_count = 0

    def _record_mt_bench_truncation(
        *,
        case_id: str,
        field: str,
        truncated: bool,
    ) -> None:
        nonlocal truncated_field_count
        if truncated and limit_event_tracker is not None:
            limit_event_tracker.record(
                "judge_input_char_truncation",
                stage="judge_input",
                field=field,
                case_id=case_id,
            )
        truncated_field_count += int(truncated)

    def _prepare_answer(answer: str, *, case_id: str, field: str) -> tuple[str, bool]:
        if not strip_thinking_before_judging:
            return answer, False
        stripped_answer, stripped = strip_thinking_tags_with_metadata(answer)
        if stripped and limit_event_tracker is not None:
            limit_event_tracker.record(
                "thinking_trace_stripped_before_judging",
                stage="judge_input",
                field=field,
                case_id=case_id,
                original_length=len(answer),
                final_length=len(stripped_answer),
            )
        return stripped_answer, stripped

    for pair_row in iter_mt_bench_pairwise_rows(
        questions=questions,
        completions_a=completions_a,
        completions_b=completions_b,
        truncate_input_chars=truncate_input_chars,
    ):
        category = pair_row.category
        if eval_single:
            case_id = f"{pair_row.question_id}:turn1"
            prompt = _select_preset_prompt(
                category,
                multi_turn=False,
                prompt_preset=prompt_preset,
                provide_explanation=provide_explanation,
            )
            answer_a, answer_a_stripped = _prepare_answer(
                pair_row.answer_a_1,
                case_id=case_id,
                field="answer_a_1",
            )
            answer_b, answer_b_stripped = _prepare_answer(
                pair_row.answer_b_1,
                case_id=case_id,
                field="answer_b_1",
            )
            _record_mt_bench_truncation(
                case_id=case_id,
                field="turn_1_question",
                truncated=pair_row.turn_1_question_truncated,
            )
            _record_mt_bench_truncation(
                case_id=case_id,
                field="answer_a_1",
                truncated=pair_row.answer_a_1_truncated,
            )
            _record_mt_bench_truncation(
                case_id=case_id,
                field="answer_b_1",
                truncated=pair_row.answer_b_1_truncated,
            )
            prompt_kwargs: dict[str, str] = {
                "question": pair_row.turn_1_question,
                "answer_a": answer_a,
                "answer_b": answer_b,
            }
            limit_flags = {
                "turn_1_question_truncated": pair_row.turn_1_question_truncated,
                "answer_a_1_truncated": pair_row.answer_a_1_truncated,
                "answer_b_1_truncated": pair_row.answer_b_1_truncated,
                "answer_a_1_reasoning_stripped": answer_a_stripped,
                "answer_b_1_reasoning_stripped": answer_b_stripped,
            }
            if prompt.ref_based:
                _record_mt_bench_truncation(
                    case_id=case_id,
                    field="ref_1",
                    truncated=pair_row.ref_1_truncated,
                )
                prompt_kwargs["ref_answer_1"] = pair_row.ref_1
                limit_flags["ref_1_truncated"] = pair_row.ref_1_truncated
            items.append(
                {
                    "case_id": case_id,
                    "question_id": pair_row.question_id,
                    "category": category,
                    "turn": 1,
                    "prompt": prompt,
                    "prompt_name": prompt.name,
                    "prompt_kwargs": prompt_kwargs,
                    "answer_fields": _answer_field_names(prompt),
                    "limit_flags": limit_flags,
                    "limit_event_tracker": limit_event_tracker,
                }
            )

        if eval_multi and pair_row.turn_2_question:
            case_id = f"{pair_row.question_id}:turn2"
            prompt = _select_preset_prompt(
                category,
                multi_turn=True,
                prompt_preset=prompt_preset,
                provide_explanation=provide_explanation,
            )
            answer_a_1, answer_a_1_stripped = _prepare_answer(
                pair_row.answer_a_1,
                case_id=case_id,
                field="answer_a_1",
            )
            answer_a_2, answer_a_2_stripped = _prepare_answer(
                pair_row.answer_a_2,
                case_id=case_id,
                field="answer_a_2",
            )
            answer_b_1, answer_b_1_stripped = _prepare_answer(
                pair_row.answer_b_1,
                case_id=case_id,
                field="answer_b_1",
            )
            answer_b_2, answer_b_2_stripped = _prepare_answer(
                pair_row.answer_b_2,
                case_id=case_id,
                field="answer_b_2",
            )
            for field, truncated in (
                ("turn_1_question", pair_row.turn_1_question_truncated),
                ("turn_2_question", pair_row.turn_2_question_truncated),
                ("answer_a_1", pair_row.answer_a_1_truncated),
                ("answer_a_2", pair_row.answer_a_2_truncated),
                ("answer_b_1", pair_row.answer_b_1_truncated),
                ("answer_b_2", pair_row.answer_b_2_truncated),
            ):
                _record_mt_bench_truncation(
                    case_id=case_id,
                    field=field,
                    truncated=truncated,
                )
            prompt_kwargs = {
                "question_1": pair_row.turn_1_question,
                "question_2": pair_row.turn_2_question,
                "answer_a_1": answer_a_1,
                "answer_a_2": answer_a_2,
                "answer_b_1": answer_b_1,
                "answer_b_2": answer_b_2,
            }
            limit_flags = {
                "turn_1_question_truncated": pair_row.turn_1_question_truncated,
                "turn_2_question_truncated": pair_row.turn_2_question_truncated,
                "answer_a_1_truncated": pair_row.answer_a_1_truncated,
                "answer_a_2_truncated": pair_row.answer_a_2_truncated,
                "answer_b_1_truncated": pair_row.answer_b_1_truncated,
                "answer_b_2_truncated": pair_row.answer_b_2_truncated,
                "answer_a_1_reasoning_stripped": answer_a_1_stripped,
                "answer_a_2_reasoning_stripped": answer_a_2_stripped,
                "answer_b_1_reasoning_stripped": answer_b_1_stripped,
                "answer_b_2_reasoning_stripped": answer_b_2_stripped,
            }
            if prompt.ref_based:
                _record_mt_bench_truncation(
                    case_id=case_id,
                    field="ref_1",
                    truncated=pair_row.ref_1_truncated,
                )
                _record_mt_bench_truncation(
                    case_id=case_id,
                    field="ref_2",
                    truncated=pair_row.ref_2_truncated,
                )
                prompt_kwargs["ref_answer_1"] = pair_row.ref_1
                prompt_kwargs["ref_answer_2"] = pair_row.ref_2
                limit_flags["ref_1_truncated"] = pair_row.ref_1_truncated
                limit_flags["ref_2_truncated"] = pair_row.ref_2_truncated
            items.append(
                {
                    "case_id": case_id,
                    "question_id": pair_row.question_id,
                    "category": category,
                    "turn": 2,
                    "prompt": prompt,
                    "prompt_name": prompt.name,
                    "prompt_kwargs": prompt_kwargs,
                    "answer_fields": _answer_field_names(prompt),
                    "limit_flags": limit_flags,
                    "limit_event_tracker": limit_event_tracker,
                }
            )
    if truncated_field_count:
        logger.warning(
            "Warning: truncated %s judge inputs to %s characters before evaluation.",
            truncated_field_count,
            truncate_input_chars,
        )
    return items


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
    strip_thinking_before_judging: bool = False,
    judge_tokenizer: Any | None = None,
    max_judge_model_len: int | None = None,
    max_out_tokens_judge: int | None = None,
    usage_tracker: OpenRouterReferencePricingTracker | None = None,
    usage_phase: str | None = None,
    limit_event_tracker: LimitEventTracker | None = None,
) -> tuple[pd.Series, list[dict[str, Any]], list[dict[str, object]], int]:
    assert turns_mode in ("both", "single", "multi")
    assert swap_mode in ("fixed", "both")

    eval_single = turns_mode in ("both", "single")
    eval_multi = turns_mode in ("both", "multi")

    items = _build_mt_bench_preset_items(
        questions=questions,
        completions_a=completions_a,
        completions_b=completions_b,
        eval_single=eval_single,
        eval_multi=eval_multi,
        truncate_input_chars=truncate_input_chars,
        prompt_preset=prompt_preset,
        provide_explanation=provide_explanation,
        strip_thinking_before_judging=strip_thinking_before_judging,
        limit_event_tracker=limit_event_tracker,
    )

    judgments, prompt_kwargs_used = _infer_by_prompt_groups(
        judge_chat_model=judge_chat_model,
        items=items,
        use_tqdm=use_tqdm,
        swap_answers=False,
        judge_tokenizer=judge_tokenizer,
        max_judge_model_len=max_judge_model_len,
        max_out_tokens_judge=max_out_tokens_judge,
        usage_tracker=usage_tracker,
        usage_phase=usage_phase,
        usage_model_spec=judge_model,
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
            prompt: MTBenchPresetPrompt = item["prompt"]
            parsed_preference = PairScore(
                parser_mode=prompt.parser_mode
            ).parse_model_raw(raw_judgment)
            normalized_preference = _normalize_preference(
                parsed_preference,
                swapped=swapped,
            )
            annotation_row = {
                "question_id": item["question_id"],
                "category": item["category"],
                "turn": item["turn"],
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
            annotation_row.update(item.get("limit_flags", {}))
            annotations.append(annotation_row)
            metadata.append(
                {
                    "question_id": item["question_id"],
                    "category": item["category"],
                    "turn": item["turn"],
                }
            )
            preferences.append(normalized_preference)

    _append_results(judgments, prompt_kwargs_used, swapped=False)

    if swap_mode == "both":
        swapped_judgments, swapped_prompt_kwargs = _infer_by_prompt_groups(
            judge_chat_model=judge_chat_model,
            items=items,
            use_tqdm=use_tqdm,
            swap_answers=True,
            judge_tokenizer=judge_tokenizer,
            max_judge_model_len=max_judge_model_len,
            max_out_tokens_judge=max_out_tokens_judge,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
            usage_model_spec=judge_model,
        )
        _append_results(swapped_judgments, swapped_prompt_kwargs, swapped=True)

    return pd.Series(preferences, dtype=float), annotations, metadata, 0
