"""MT-Bench pairwise judging aligned with FastChat ``llm_judge`` (``data/judge_prompts.jsonl``)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from judgearena.judge_prompt_presets import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    SKYWORK_JUDGE_PROMPT_PRESET,
)
from judgearena.mt_bench.common import (
    MT_BENCH_REFERENCE_CATEGORIES,
    iter_mt_bench_pairwise_rows,
)
from judgearena.mt_bench.prompt_templates import (
    build_mt_bench_user_prompt_template,
    render_mt_bench_prompt_text,
)
from judgearena.openrouter_reference_pricing import OpenRouterReferencePricingTracker
from judgearena.utils import (
    LimitEventTracker,
    do_inference,
    strip_thinking_tags,
    strip_thinking_tags_with_metadata,
)

FASTCHAT_TEMPERATURE_CONFIG: dict[str, float] = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": 0.0,
}

FastChatVerdict = Literal["A", "B", "tie", "error"]
PairwiseWinner = Literal["model_A", "model_B", "tie", "error"]


@dataclass(frozen=True)
class FastChatPairwisePrompt:
    name: str
    system_prompt: str
    user_prompt_template: str
    multi_turn: bool
    ref_based: bool


_SYSTEM_BASE_FILE = "system-base.txt"


def _build_system_prompt(
    *,
    user_subject: str,
    task_description: str,
    begin_instruction: str,
    focus_line: str = "",
) -> str:
    focus_segment = f"{focus_line} " if focus_line else ""
    return render_mt_bench_prompt_text(
        _SYSTEM_BASE_FILE,
        user_subject=user_subject,
        task_description=task_description,
        focus_line=focus_segment,
        begin_instruction=begin_instruction,
    )


def _load_pairwise_prompt(
    *,
    name: str,
    multi_turn: bool,
    ref_based: bool,
    system_user_subject: str,
    system_task_description: str,
    system_begin_instruction: str,
    system_focus_line: str = "",
) -> FastChatPairwisePrompt:
    return FastChatPairwisePrompt(
        name=name,
        multi_turn=multi_turn,
        ref_based=ref_based,
        system_prompt=_build_system_prompt(
            user_subject=system_user_subject,
            task_description=system_task_description,
            begin_instruction=system_begin_instruction,
            focus_line=system_focus_line,
        ),
        user_prompt_template=build_mt_bench_user_prompt_template(
            multi_turn=multi_turn,
            ref_based=ref_based,
        ),
    )


_PAIR_V2 = _load_pairwise_prompt(
    name="pair-v2",
    multi_turn=False,
    ref_based=False,
    system_user_subject="question displayed below",
    system_task_description=(
        "You should choose the assistant that follows the user's instructions and answers "
        "the user's question better. Your evaluation should consider factors such as the "
        "helpfulness, relevance, accuracy, depth, creativity, and level of detail of their "
        "responses."
    ),
    system_begin_instruction="comparing the two responses and provide a short explanation",
)

_PAIR_V2_MULTI = _load_pairwise_prompt(
    name="pair-v2-multi-turn",
    multi_turn=True,
    ref_based=False,
    system_user_subject="questions",
    system_task_description=(
        "You should choose the assistant that follows the user's instructions and answers "
        "the user's questions better. Your evaluation should consider factors such as the "
        "helpfulness, relevance, accuracy, depth, creativity, and level of detail of their "
        "responses."
    ),
    system_focus_line="You should focus on who provides a better answer to the second user question.",
    system_begin_instruction=(
        "comparing the responses of the two assistants and provide a short explanation"
    ),
)

_PAIR_MATH_V1 = _load_pairwise_prompt(
    name="pair-math-v1",
    multi_turn=False,
    ref_based=True,
    system_user_subject="question displayed below",
    system_task_description=(
        "Your evaluation should consider correctness and helpfulness. You will be given a "
        "reference answer, assistant A's answer, and assistant B's answer. Your job is to "
        "evaluate which assistant's answer is better."
    ),
    system_begin_instruction=(
        "comparing both assistants' answers with the reference answer. Identify and correct any mistakes"
    ),
)

_PAIR_MATH_V1_MULTI = _load_pairwise_prompt(
    name="pair-math-v1-multi-turn",
    multi_turn=True,
    ref_based=True,
    system_user_subject="questions",
    system_task_description=(
        "Your evaluation should consider correctness and helpfulness. You will be given "
        "reference answers, the assistant A's answers, the assistant B's answers. Your job is "
        "to determine which assistant provides correct and helpful answers to the second user question."
    ),
    system_begin_instruction=(
        "comparing both assistants' answers with the reference answers. Identify and correct any mistakes"
    ),
)


_SKYWORK_PAIR_V2 = _load_pairwise_prompt(
    name="skywork-pair-v2",
    multi_turn=False,
    ref_based=False,
    system_user_subject="prompt displayed below",
    system_task_description=(
        "You should choose the assistant that follows the user's instructions and "
        "answers the user's prompt better. Your evaluation should consider factors "
        "such as helpfulness, relevance, accuracy, depth, creativity, and level "
        "of detail of the responses."
    ),
    system_begin_instruction="carefully comparing the two responses",
)

_SKYWORK_PAIR_V2_MULTI = _load_pairwise_prompt(
    name="skywork-pair-v2-multi-turn",
    multi_turn=True,
    ref_based=False,
    system_user_subject="questions",
    system_task_description=(
        "You should choose the assistant that follows the user's instructions and "
        "answers the user's questions better. Your evaluation should consider "
        "factors such as helpfulness, relevance, accuracy, depth, creativity, and "
        "level of detail of the responses."
    ),
    system_focus_line=(
        "You should focus on which assistant better answers the second user question."
    ),
    system_begin_instruction="carefully comparing the two conversations",
)

_SKYWORK_PAIR_MATH_V1 = _load_pairwise_prompt(
    name="skywork-pair-math-v1",
    multi_turn=False,
    ref_based=True,
    system_user_subject="prompt displayed below",
    system_task_description=(
        "You will be given a reference answer, assistant A's answer, and "
        "assistant B's answer. Your evaluation should focus on correctness and "
        "helpfulness while deciding which assistant is better."
    ),
    system_begin_instruction="carefully comparing both assistants' answers with the reference answer",
)

_SKYWORK_PAIR_MATH_V1_MULTI = _load_pairwise_prompt(
    name="skywork-pair-math-v1-multi-turn",
    multi_turn=True,
    ref_based=True,
    system_user_subject="questions",
    system_task_description=(
        "You will be given reference answers together with assistant A's and "
        "assistant B's answers. Your evaluation should focus on correctness and "
        "helpfulness while deciding which assistant better answers the second user question."
    ),
    system_begin_instruction="carefully comparing both assistants' answers with the reference answers",
)

_FASTCHAT_PROMPT_PRESET_REGISTRY: dict[str, dict[str, FastChatPairwisePrompt]] = {
    DEFAULT_JUDGE_PROMPT_PRESET: {
        "single": _PAIR_V2,
        "multi": _PAIR_V2_MULTI,
        "single_ref": _PAIR_MATH_V1,
        "multi_ref": _PAIR_MATH_V1_MULTI,
    },
    SKYWORK_JUDGE_PROMPT_PRESET: {
        "single": _SKYWORK_PAIR_V2,
        "multi": _SKYWORK_PAIR_V2_MULTI,
        "single_ref": _SKYWORK_PAIR_MATH_V1,
        "multi_ref": _SKYWORK_PAIR_MATH_V1_MULTI,
    },
}


def _parse_fastchat_verdict(judgment: str) -> FastChatVerdict:
    stripped = strip_thinking_tags(judgment).strip()
    if "[[A]]" in stripped:
        return "A"
    if "[[B]]" in stripped:
        return "B"
    if "[[C]]" in stripped:
        return "tie"
    return "error"


def _map_verdict_to_winner(verdict: FastChatVerdict, swapped: bool) -> PairwiseWinner:
    if verdict == "tie":
        return "tie"
    if verdict == "error":
        return "error"
    if verdict == "A":
        return "model_B" if swapped else "model_A"
    if verdict == "B":
        return "model_A" if swapped else "model_B"
    return "error"


def _conservative_winner(
    g1: PairwiseWinner, g2: PairwiseWinner
) -> tuple[PairwiseWinner, bool]:
    """Conservative position-bias handling (FastChat/MT-Bench paper).

    Declare a winner only if the two orderings agree; otherwise treat as tie.
    """
    if g1 == "error" or g2 == "error":
        return "error", False
    if g1 == g2:
        return g1, False
    return "tie", True


def _winner_to_preference(winner: PairwiseWinner) -> float:
    if winner == "model_A":
        return 0.0
    if winner == "model_B":
        return 1.0
    if winner == "tie":
        return 0.5
    return math.nan


def _select_prompt(
    category: str | None,
    multi_turn: bool,
    *,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
) -> FastChatPairwisePrompt:
    prompt_variants = _FASTCHAT_PROMPT_PRESET_REGISTRY.get(prompt_preset)
    if prompt_variants is None:
        supported = ", ".join(sorted(_FASTCHAT_PROMPT_PRESET_REGISTRY))
        raise ValueError(
            f"Unsupported MT-Bench prompt preset '{prompt_preset}'. Choose from: {supported}."
        )
    needs_ref = (category or "") in MT_BENCH_REFERENCE_CATEGORIES
    if needs_ref and multi_turn:
        return prompt_variants["multi_ref"]
    if needs_ref:
        return prompt_variants["single_ref"]
    if multi_turn:
        return prompt_variants["multi"]
    return prompt_variants["single"]


def _group_indices_by_prompt(
    items: list[dict[str, Any]],
) -> dict[str, list[int]]:
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


def _infer_by_prompt_groups(
    *,
    judge_chat_model,
    items: list[dict[str, Any]],
    use_tqdm: bool,
    swap_answers: bool,
    usage_tracker: OpenRouterReferencePricingTracker | None = None,
    usage_phase: str | None = None,
    usage_model_spec: str | None = None,
) -> list[str]:
    """Run judge inference, grouping by prompt variant for batching."""
    grouped_indices = _group_indices_by_prompt(items)

    judgments: list[str] = [""] * len(items)
    for _prompt_name, idxs in grouped_indices.items():
        prompt: FastChatPairwisePrompt = items[idxs[0]]["prompt"]
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompt.system_prompt), ("user", prompt.user_prompt_template)]
        )

        batch_kwargs = []
        for i in idxs:
            kwargs = items[i]["prompt_kwargs"]
            if swap_answers:
                kwargs = _swap_prompt_kwargs(kwargs, multi_turn=prompt.multi_turn)
            batch_kwargs.append(kwargs)

        prompt_inputs = prompt_template.batch(batch_kwargs)
        outs = do_inference(
            chat_model=judge_chat_model,
            inputs=prompt_inputs,
            use_tqdm=use_tqdm,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
            usage_model_spec=usage_model_spec,
        )
        for i, out in zip(idxs, outs, strict=True):
            judgments[i] = str(out)
    return judgments


def _build_fastchat_judge_items(
    *,
    questions: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    eval_single: bool,
    eval_multi: bool,
    truncate_input_chars: int | None,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    strip_thinking_before_judging: bool = False,
    limit_event_tracker: LimitEventTracker | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def _record_mt_bench_truncation(
        *, case_id: str, field: str, truncated: bool
    ) -> None:
        if truncated and limit_event_tracker is not None:
            limit_event_tracker.record(
                "mt_bench_field_char_truncation",
                stage="judge_input",
                field=field,
                case_id=case_id,
            )

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
            prompt = _select_prompt(
                category, multi_turn=False, prompt_preset=prompt_preset
            )
            answer_a, answer_a_stripped = _prepare_answer(
                pair_row.answer_a_1, case_id=case_id, field="answer_a_1"
            )
            answer_b, answer_b_stripped = _prepare_answer(
                pair_row.answer_b_1, case_id=case_id, field="answer_b_1"
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
            kwargs: dict[str, str] = {
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
                kwargs["ref_answer_1"] = pair_row.ref_1
                limit_flags["ref_1_truncated"] = pair_row.ref_1_truncated
            items.append(
                {
                    "question_id": pair_row.question_id,
                    "category": category,
                    "turn": 1,
                    "prompt": prompt,
                    "prompt_name": prompt.name,
                    "prompt_kwargs": kwargs,
                    "limit_flags": limit_flags,
                }
            )

        if eval_multi and pair_row.turn_2_question:
            case_id = f"{pair_row.question_id}:turn2"
            prompt = _select_prompt(
                category, multi_turn=True, prompt_preset=prompt_preset
            )
            answer_a_1, answer_a_1_stripped = _prepare_answer(
                pair_row.answer_a_1, case_id=case_id, field="answer_a_1"
            )
            answer_a_2, answer_a_2_stripped = _prepare_answer(
                pair_row.answer_a_2, case_id=case_id, field="answer_a_2"
            )
            answer_b_1, answer_b_1_stripped = _prepare_answer(
                pair_row.answer_b_1, case_id=case_id, field="answer_b_1"
            )
            answer_b_2, answer_b_2_stripped = _prepare_answer(
                pair_row.answer_b_2, case_id=case_id, field="answer_b_2"
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
                    case_id=case_id, field=field, truncated=truncated
                )
            kwargs = {
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
                kwargs["ref_answer_1"] = pair_row.ref_1
                kwargs["ref_answer_2"] = pair_row.ref_2
                limit_flags["ref_1_truncated"] = pair_row.ref_1_truncated
                limit_flags["ref_2_truncated"] = pair_row.ref_2_truncated
            items.append(
                {
                    "question_id": pair_row.question_id,
                    "category": category,
                    "turn": 2,
                    "prompt": prompt,
                    "prompt_name": prompt.name,
                    "prompt_kwargs": kwargs,
                    "limit_flags": limit_flags,
                }
            )
    return items


def _resolve_fastchat_item_result(
    *,
    item: dict[str, Any],
    g1_raw: str,
    g2_raw: str | None,
    judge_model: str,
    model_a: str,
    model_b: str,
) -> tuple[dict[str, Any], dict[str, object], float, bool]:
    prompt: FastChatPairwisePrompt = item["prompt"]
    kwargs = item["prompt_kwargs"]
    g1_user_prompt = prompt.user_prompt_template.format(**kwargs)
    g1_verdict = _parse_fastchat_verdict(g1_raw)
    g1_winner = _map_verdict_to_winner(g1_verdict, swapped=False)

    final_winner = g1_winner
    inconsistent = False
    annotation_row: dict[str, Any] = {
        "question_id": item["question_id"],
        "category": item["category"],
        "turn": item["turn"],
        "model_A": model_a,
        "model_B": model_b,
        "judge": judge_model,
        "prompt_name": prompt.name,
        "system_prompt": prompt.system_prompt,
        "g1_user_prompt": g1_user_prompt,
        "g1_judgment": g1_raw,
        "g1_verdict": g1_verdict,
        "g1_winner": g1_winner,
    }
    annotation_row.update(item.get("limit_flags", {}))

    if g2_raw is not None:
        g2_verdict = _parse_fastchat_verdict(g2_raw)
        g2_winner = _map_verdict_to_winner(g2_verdict, swapped=True)
        final_winner, inconsistent = _conservative_winner(g1_winner, g2_winner)
        annotation_row.update(
            {
                "g2_user_prompt": prompt.user_prompt_template.format(
                    **_swap_prompt_kwargs(kwargs, multi_turn=prompt.multi_turn)
                ),
                "g2_judgment": g2_raw,
                "g2_verdict": g2_verdict,
                "g2_winner": g2_winner,
                "final_winner": final_winner,
                "inconsistent": inconsistent,
            }
        )
    else:
        annotation_row["final_winner"] = final_winner
        annotation_row["inconsistent"] = False

    preference = _winner_to_preference(final_winner)
    annotation_row["preference"] = preference
    metadata = {
        "question_id": item["question_id"],
        "category": item["category"],
        "turn": item["turn"],
    }
    return annotation_row, metadata, preference, inconsistent


def judge_mt_bench_pairwise_fastchat(
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
    strip_thinking_before_judging: bool = False,
    usage_tracker: OpenRouterReferencePricingTracker | None = None,
    usage_phase: str | None = None,
    limit_event_tracker: LimitEventTracker | None = None,
) -> tuple[pd.Series, list[dict[str, Any]], list[dict[str, object]], int]:
    """Run FastChat-style MT-Bench pairwise judging with bracketed verdict outputs."""
    assert turns_mode in ("both", "single", "multi")
    assert swap_mode in ("fixed", "both")

    eval_single = turns_mode in ("both", "single")
    eval_multi = turns_mode in ("both", "multi")

    items = _build_fastchat_judge_items(
        questions=questions,
        completions_a=completions_a,
        completions_b=completions_b,
        eval_single=eval_single,
        eval_multi=eval_multi,
        truncate_input_chars=truncate_input_chars,
        prompt_preset=prompt_preset,
        strip_thinking_before_judging=strip_thinking_before_judging,
        limit_event_tracker=limit_event_tracker,
    )

    g1_judgments = _infer_by_prompt_groups(
        judge_chat_model=judge_chat_model,
        items=items,
        use_tqdm=use_tqdm,
        swap_answers=False,
        usage_tracker=usage_tracker,
        usage_phase=usage_phase,
        usage_model_spec=judge_model,
    )

    g2_judgments: list[str] | None = None
    if swap_mode == "both":
        g2_judgments = _infer_by_prompt_groups(
            judge_chat_model=judge_chat_model,
            items=items,
            use_tqdm=use_tqdm,
            swap_answers=True,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
            usage_model_spec=judge_model,
        )

    annotations: list[dict[str, Any]] = []
    metadata: list[dict[str, object]] = []
    prefs: list[float] = []
    num_inconsistent = 0

    for idx, item in enumerate(items):
        g2_raw = g2_judgments[idx] if g2_judgments is not None else None
        annotation_row, item_metadata, preference, inconsistent = (
            _resolve_fastchat_item_result(
                item=item,
                g1_raw=g1_judgments[idx],
                g2_raw=g2_raw,
                judge_model=judge_model,
                model_a=model_a,
                model_b=model_b,
            )
        )
        if inconsistent:
            num_inconsistent += 1
        annotations.append(annotation_row)
        metadata.append(item_metadata)
        prefs.append(preference)

    return pd.Series(prefs, dtype=float), annotations, metadata, num_inconsistent
