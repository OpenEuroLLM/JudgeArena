"""Rendering for the official WildBench V2 prompts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from judgearena.utils import truncate

EMPTY_RESPONSE = "[This model response is empty.]"


@dataclass(frozen=True)
class RenderedRewardPrompt:
    """Pairwise prompt plus the exact response text visible to the judge."""

    text: str
    candidate_output: str
    baseline_output: str


def _prompt_field(
    value: object, *, max_words: int | None, max_chars: int | None
) -> str:
    text = "" if value is None else str(value)
    if max_words is not None and len(text.split(" ")) > max_words:
        text = " ".join(text.split(" ")[:max_words]) + "... (truncated)"
    return truncate(text, max_len=max_chars)


def render_score_prompt(
    template: str,
    example: pd.Series,
    model_output: str,
    *,
    max_words: int | None,
    max_chars: int | None,
) -> str:
    """Fill the four fields used by the upstream WB-Score template."""
    checklist = "".join(f"- {item}\n" for item in example["checklist"])
    replacements = {
        "{$history}": _prompt_field(
            example["history"], max_words=max_words, max_chars=max_chars
        ),
        "{$user_query}": _prompt_field(
            example["instruction"], max_words=max_words, max_chars=max_chars
        ),
        "{$model_output}": _prompt_field(
            model_output, max_words=max_words, max_chars=max_chars
        ),
        "{$checklist}": checklist,
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def render_reward_prompt(
    template: str,
    example: pd.Series,
    candidate_output: str,
    baseline_output: str,
    *,
    candidate_is_a: bool,
    max_words: int,
    max_chars: int | None,
) -> RenderedRewardPrompt:
    """Fill the official WB-Reward template in one deterministic orientation."""
    candidate_visible = _prompt_field(
        candidate_output, max_words=max_words, max_chars=max_chars
    )
    baseline_visible = _prompt_field(
        baseline_output, max_words=max_words, max_chars=max_chars
    )
    if not candidate_visible.strip():
        candidate_visible = EMPTY_RESPONSE
    if not baseline_visible.strip():
        baseline_visible = EMPTY_RESPONSE
    output_a, output_b = (
        (candidate_visible, baseline_visible)
        if candidate_is_a
        else (baseline_visible, candidate_visible)
    )
    checklist = "".join(f"- {item}\n" for item in example["checklist"])
    replacements = {
        "{$history}": _prompt_field(
            example["history"], max_words=max_words, max_chars=max_chars
        ),
        "{$user_query}": _prompt_field(
            example["instruction"], max_words=max_words, max_chars=max_chars
        ),
        "{$candidate_A}": output_a,
        "{$candidate_B}": output_b,
        "{$checklist}": checklist,
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return RenderedRewardPrompt(
        text=rendered,
        candidate_output=candidate_visible,
        baseline_output=baseline_visible,
    )
