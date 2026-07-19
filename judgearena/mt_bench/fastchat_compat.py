"""MT-Bench pairwise judging aligned with FastChat ``llm_judge`` (``data/judge_prompts.jsonl``)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from judgearena.mt_bench.common import (
    is_reference_based_category,
    resolve_mt_bench_turn_flags,
)
from judgearena.mt_bench.pairwise_judging import (
    MTBenchJudgeItem,
    build_mt_bench_pairwise_judge_items,
    infer_pairwise_judgments_by_prompt_groups,
    swap_pairwise_answer_kwargs,
)
from judgearena.mt_bench.prompt_templates import (
    build_mt_bench_user_prompt_template,
    render_mt_bench_prompt_text,
)
from judgearena.prompts.registry import DEFAULT_JUDGE_PROMPT_PRESET
from judgearena.utils import strip_thinking_tags

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


_FASTCHAT_PROMPT_PRESET_REGISTRY: dict[str, dict[str, FastChatPairwisePrompt]] = {
    DEFAULT_JUDGE_PROMPT_PRESET: {
        "single": _PAIR_V2,
        "multi": _PAIR_V2_MULTI,
        "single_ref": _PAIR_MATH_V1,
        "multi_ref": _PAIR_MATH_V1_MULTI,
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
    needs_ref = is_reference_based_category(category)
    if needs_ref and multi_turn:
        return prompt_variants["multi_ref"]
    if needs_ref:
        return prompt_variants["single_ref"]
    if multi_turn:
        return prompt_variants["multi"]
    return prompt_variants["single"]


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
) -> list[MTBenchJudgeItem]:
    return build_mt_bench_pairwise_judge_items(
        questions=questions,
        completions_a=completions_a,
        completions_b=completions_b,
        eval_single=eval_single,
        eval_multi=eval_multi,
        truncate_input_chars=truncate_input_chars,
        select_prompt=lambda category, multi_turn: _select_prompt(
            category,
            multi_turn=multi_turn,
            prompt_preset=prompt_preset,
        ),
        strip_thinking_before_judging=strip_thinking_before_judging,
    )


def _resolve_fastchat_item_result(
    *,
    item: MTBenchJudgeItem,
    g1_raw: str,
    g2_raw: str | None,
    judge_model: str,
    model_a: str,
    model_b: str,
) -> tuple[dict[str, Any], dict[str, object], float, bool]:
    prompt: FastChatPairwisePrompt = item.prompt
    kwargs = item.prompt_kwargs
    g1_user_prompt = prompt.user_prompt_template.format(**kwargs)
    g1_verdict = _parse_fastchat_verdict(g1_raw)
    g1_winner = _map_verdict_to_winner(g1_verdict, swapped=False)

    final_winner = g1_winner
    inconsistent = False
    annotation_row: dict[str, Any] = {
        "question_id": item.question_id,
        "category": item.category,
        "turn": item.turn,
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
        # swap_mode="both": conservative agreement — keep a winner only if both
        # orderings agree, otherwise tie (FastChat/MT-Bench consistency).
        g2_verdict = _parse_fastchat_verdict(g2_raw)
        g2_winner = _map_verdict_to_winner(g2_verdict, swapped=True)
        final_winner, inconsistent = _conservative_winner(g1_winner, g2_winner)
        annotation_row.update(
            {
                "g2_user_prompt": prompt.user_prompt_template.format(
                    **swap_pairwise_answer_kwargs(
                        kwargs,
                        multi_turn=prompt.multi_turn,
                    )
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
        "question_id": item.question_id,
        "category": item.category,
        "turn": item.turn,
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
) -> tuple[pd.Series, list[dict[str, Any]], list[dict[str, object]], int]:
    """Run FastChat-style MT-Bench pairwise judging with bracketed verdict outputs."""
    assert swap_mode in ("fixed", "both")
    eval_single, eval_multi = resolve_mt_bench_turn_flags(turns_mode)

    items = _build_fastchat_judge_items(
        questions=questions,
        completions_a=completions_a,
        completions_b=completions_b,
        eval_single=eval_single,
        eval_multi=eval_multi,
        truncate_input_chars=truncate_input_chars,
        prompt_preset=prompt_preset,
        strip_thinking_before_judging=strip_thinking_before_judging,
    )

    g1_judgments, _ = infer_pairwise_judgments_by_prompt_groups(
        judge_chat_model=judge_chat_model,
        items=items,
        use_tqdm=use_tqdm,
        swap_answers=False,
    )

    g2_judgments: list[str] | None = None
    if swap_mode == "both":
        g2_judgments, _ = infer_pairwise_judgments_by_prompt_groups(
            judge_chat_model=judge_chat_model,
            items=items,
            use_tqdm=use_tqdm,
            swap_answers=True,
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
