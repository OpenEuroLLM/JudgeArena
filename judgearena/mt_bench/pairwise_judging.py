from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from judgearena.models import do_inference
from judgearena.mt_bench.common import iter_mt_bench_pairwise_rows
from judgearena.utils import strip_thinking_tags


class MTBenchPairwisePrompt(Protocol):
    name: str
    system_prompt: str | None
    user_prompt_template: str
    multi_turn: bool
    ref_based: bool


@dataclass(frozen=True)
class MTBenchJudgeItem:
    question_id: object
    category: str | None
    turn: int
    prompt: MTBenchPairwisePrompt
    prompt_kwargs: dict[str, str]

    @property
    def prompt_name(self) -> str:
        return self.prompt.name


def group_indices_by_prompt_key(items: list[MTBenchJudgeItem]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        grouped.setdefault(item.prompt_name, []).append(idx)
    return grouped


def swap_pairwise_answer_kwargs(
    kwargs: dict[str, str],
    *,
    multi_turn: bool,
) -> dict[str, str]:
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


def build_pairwise_chat_prompt_template(
    prompt: MTBenchPairwisePrompt,
) -> ChatPromptTemplate:
    message_templates: list[tuple[str, str]] = []
    if prompt.system_prompt is not None:
        message_templates.append(("system", prompt.system_prompt))
    message_templates.append(("user", prompt.user_prompt_template))
    return ChatPromptTemplate.from_messages(message_templates)


def infer_pairwise_judgments_by_prompt_groups(
    *,
    judge_chat_model,
    items: list[MTBenchJudgeItem],
    use_tqdm: bool,
    swap_answers: bool,
) -> tuple[list[str], list[dict[str, str]]]:
    judgments: list[str] = [""] * len(items)
    used_prompt_kwargs: list[dict[str, str]] = [{} for _ in items]
    for idxs in group_indices_by_prompt_key(items).values():
        prompt = items[idxs[0]].prompt
        prompt_template = build_pairwise_chat_prompt_template(prompt)
        batch_kwargs: list[dict[str, str]] = []
        for item_index in idxs:
            prompt_kwargs = dict(items[item_index].prompt_kwargs)
            if swap_answers:
                prompt_kwargs = swap_pairwise_answer_kwargs(
                    prompt_kwargs,
                    multi_turn=prompt.multi_turn,
                )
            batch_kwargs.append(prompt_kwargs)
        prompt_inputs = prompt_template.batch(batch_kwargs)
        outputs = do_inference(
            chat_model=judge_chat_model,
            inputs=prompt_inputs,
            use_tqdm=use_tqdm,
        )
        for item_index, output, prompt_kwargs in zip(
            idxs, outputs, batch_kwargs, strict=True
        ):
            judgments[item_index] = str(output)
            used_prompt_kwargs[item_index] = prompt_kwargs
    return judgments, used_prompt_kwargs


def build_mt_bench_pairwise_judge_items(
    *,
    questions: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    eval_single: bool,
    eval_multi: bool,
    truncate_input_chars: int | None,
    select_prompt: Callable[[str | None, bool], MTBenchPairwisePrompt],
    strip_thinking_before_judging: bool = False,
) -> list[MTBenchJudgeItem]:
    def _answer(text: str) -> str:
        return strip_thinking_tags(text) if strip_thinking_before_judging else text

    items: list[MTBenchJudgeItem] = []
    for pair_row in iter_mt_bench_pairwise_rows(
        questions=questions,
        completions_a=completions_a,
        completions_b=completions_b,
        truncate_input_chars=truncate_input_chars,
    ):
        category = pair_row.category
        if eval_single:
            prompt = select_prompt(category, False)
            prompt_kwargs: dict[str, str] = {
                "question": pair_row.turn_1_question,
                "answer_a": _answer(pair_row.answer_a_1),
                "answer_b": _answer(pair_row.answer_b_1),
            }
            if prompt.ref_based:
                prompt_kwargs["ref_answer_1"] = pair_row.ref_1
            items.append(
                MTBenchJudgeItem(
                    question_id=pair_row.question_id,
                    category=category,
                    turn=1,
                    prompt=prompt,
                    prompt_kwargs=prompt_kwargs,
                )
            )

        if eval_multi and pair_row.turn_2_question:
            prompt = select_prompt(category, True)
            prompt_kwargs = {
                "question_1": pair_row.turn_1_question,
                "question_2": pair_row.turn_2_question,
                "answer_a_1": _answer(pair_row.answer_a_1),
                "answer_a_2": _answer(pair_row.answer_a_2),
                "answer_b_1": _answer(pair_row.answer_b_1),
                "answer_b_2": _answer(pair_row.answer_b_2),
            }
            if prompt.ref_based:
                prompt_kwargs["ref_answer_1"] = pair_row.ref_1
                prompt_kwargs["ref_answer_2"] = pair_row.ref_2
            items.append(
                MTBenchJudgeItem(
                    question_id=pair_row.question_id,
                    category=category,
                    turn=2,
                    prompt=prompt,
                    prompt_kwargs=prompt_kwargs,
                )
            )
    return items
