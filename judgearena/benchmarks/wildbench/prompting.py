"""Rendering for the official WildBench V2 prompts."""

from __future__ import annotations

import pandas as pd

from judgearena.utils import truncate


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
