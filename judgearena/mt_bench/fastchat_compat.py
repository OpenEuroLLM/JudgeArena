"""MT-Bench pairwise judging aligned with FastChat ``llm_judge`` (``data/judge_prompts.jsonl``)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from judgearena.mt_bench.common import iter_mt_bench_pairwise_rows
from judgearena.openrouter_reference_pricing import OpenRouterReferencePricingTracker
from judgearena.utils import do_inference, strip_thinking_tags

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

FASTCHAT_NEED_REF_CATS: set[str] = {
    "math",
    "reasoning",
    "coding",
    "arena-hard-200",
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


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "mt_bench"
_SYSTEM_BASE_FILE = "system-base.txt"
_USER_SINGLE_BASE_FILE = "user-single-base.txt"
_USER_MULTI_BASE_FILE = "user-multi-base.txt"
_USER_SINGLE_REF_BLOCK_FILE = "user-single-reference-block.txt"
_USER_MULTI_REF_BLOCK_FILE = "user-multi-reference-block.txt"


def _load_prompt_text(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


def _render_prompt_text(filename: str, **kwargs: str) -> str:
    return _load_prompt_text(filename).format(**kwargs)


def _build_system_prompt(
    *,
    user_subject: str,
    task_description: str,
    begin_instruction: str,
    focus_line: str = "",
) -> str:
    focus_segment = f"{focus_line} " if focus_line else ""
    return _render_prompt_text(
        _SYSTEM_BASE_FILE,
        user_subject=user_subject,
        task_description=task_description,
        focus_line=focus_segment,
        begin_instruction=begin_instruction,
        output_format_instruction=(
            "After providing your explanation, output your final verdict by strictly following this format: "
            '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better, '
            'and "[[C]]" for a tie.'
        ),
    )


def _build_user_prompt_template(*, multi_turn: bool, ref_based: bool) -> str:
    base_filename = _USER_MULTI_BASE_FILE if multi_turn else _USER_SINGLE_BASE_FILE
    reference_block = ""
    if ref_based:
        ref_block_filename = (
            _USER_MULTI_REF_BLOCK_FILE if multi_turn else _USER_SINGLE_REF_BLOCK_FILE
        )
        reference_block = _load_prompt_text(ref_block_filename).rstrip("\n") + "\n\n"
    return _render_prompt_text(base_filename, reference_block=reference_block)


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
        user_prompt_template=_build_user_prompt_template(
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


def _select_prompt(category: str | None, multi_turn: bool) -> FastChatPairwisePrompt:
    needs_ref = (category or "") in FASTCHAT_NEED_REF_CATS
    if needs_ref and multi_turn:
        return _PAIR_MATH_V1_MULTI
    if needs_ref:
        return _PAIR_MATH_V1
    if multi_turn:
        return _PAIR_V2_MULTI
    return _PAIR_V2


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
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for pair_row in iter_mt_bench_pairwise_rows(
        questions=questions,
        completions_a=completions_a,
        completions_b=completions_b,
        truncate_input_chars=truncate_input_chars,
    ):
        category = pair_row.category
        if eval_single:
            prompt = _select_prompt(category, multi_turn=False)
            kwargs: dict[str, str] = {
                "question": pair_row.turn_1_question,
                "answer_a": pair_row.answer_a_1,
                "answer_b": pair_row.answer_b_1,
            }
            if prompt.ref_based:
                kwargs["ref_answer_1"] = pair_row.ref_1
            items.append(
                {
                    "question_id": pair_row.question_id,
                    "category": category,
                    "turn": 1,
                    "prompt": prompt,
                    "prompt_name": prompt.name,
                    "prompt_kwargs": kwargs,
                }
            )

        if eval_multi and pair_row.turn_2_question:
            prompt = _select_prompt(category, multi_turn=True)
            kwargs = {
                "question_1": pair_row.turn_1_question,
                "question_2": pair_row.turn_2_question,
                "answer_a_1": pair_row.answer_a_1,
                "answer_a_2": pair_row.answer_a_2,
                "answer_b_1": pair_row.answer_b_1,
                "answer_b_2": pair_row.answer_b_2,
            }
            if prompt.ref_based:
                kwargs["ref_answer_1"] = pair_row.ref_1
                kwargs["ref_answer_2"] = pair_row.ref_2
            items.append(
                {
                    "question_id": pair_row.question_id,
                    "category": category,
                    "turn": 2,
                    "prompt": prompt,
                    "prompt_name": prompt.name,
                    "prompt_kwargs": kwargs,
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
    usage_tracker: OpenRouterReferencePricingTracker | None = None,
    usage_phase: str | None = None,
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
