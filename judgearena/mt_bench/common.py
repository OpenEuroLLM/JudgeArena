from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from judgearena.utils import safe_text_with_metadata


@dataclass(frozen=True)
class MTBenchPairwiseRow:
    question_id: object
    category: str | None
    turn_1_question: str
    turn_2_question: str
    answer_a_1: str
    answer_a_2: str
    answer_b_1: str
    answer_b_2: str
    ref_1: str
    ref_2: str
    turn_1_question_truncated: bool = False
    turn_2_question_truncated: bool = False
    answer_a_1_truncated: bool = False
    answer_a_2_truncated: bool = False
    answer_b_1_truncated: bool = False
    answer_b_2_truncated: bool = False
    ref_1_truncated: bool = False
    ref_2_truncated: bool = False


def iter_mt_bench_pairwise_rows(
    *,
    questions: pd.DataFrame,
    completions_a: pd.DataFrame,
    completions_b: pd.DataFrame,
    truncate_input_chars: int | None,
) -> Iterator[MTBenchPairwiseRow]:
    for question_id in questions.index.tolist():
        row = questions.loc[question_id]
        comp_a_row = (
            completions_a.loc[question_id]
            if question_id in completions_a.index
            else completions_a.iloc[0]
        )
        comp_b_row = (
            completions_b.loc[question_id]
            if question_id in completions_b.index
            else completions_b.iloc[0]
        )
        turn_1_question, turn_1_question_truncated = safe_text_with_metadata(
            row.get("turn_1"),
            truncate_input_chars,
        )
        turn_2_question, turn_2_question_truncated = safe_text_with_metadata(
            row.get("turn_2"),
            truncate_input_chars,
        )
        answer_a_1, answer_a_1_truncated = safe_text_with_metadata(
            comp_a_row.get("completion_turn_1", ""),
            truncate_input_chars,
        )
        answer_a_2, answer_a_2_truncated = safe_text_with_metadata(
            comp_a_row.get("completion_turn_2", ""),
            truncate_input_chars,
        )
        answer_b_1, answer_b_1_truncated = safe_text_with_metadata(
            comp_b_row.get("completion_turn_1", ""),
            truncate_input_chars,
        )
        answer_b_2, answer_b_2_truncated = safe_text_with_metadata(
            comp_b_row.get("completion_turn_2", ""),
            truncate_input_chars,
        )
        ref_1, ref_1_truncated = safe_text_with_metadata(
            row.get("reference_turn_1"),
            truncate_input_chars,
        )
        ref_2, ref_2_truncated = safe_text_with_metadata(
            row.get("reference_turn_2"),
            truncate_input_chars,
        )
        yield MTBenchPairwiseRow(
            question_id=question_id,
            category=row.get("category"),
            turn_1_question=turn_1_question,
            turn_2_question=turn_2_question,
            answer_a_1=answer_a_1,
            answer_a_2=answer_a_2,
            answer_b_1=answer_b_1,
            answer_b_2=answer_b_2,
            ref_1=ref_1,
            ref_2=ref_2,
            turn_1_question_truncated=turn_1_question_truncated,
            turn_2_question_truncated=turn_2_question_truncated,
            answer_a_1_truncated=answer_a_1_truncated,
            answer_a_2_truncated=answer_a_2_truncated,
            answer_b_1_truncated=answer_b_1_truncated,
            answer_b_2_truncated=answer_b_2_truncated,
            ref_1_truncated=ref_1_truncated,
            ref_2_truncated=ref_2_truncated,
        )
