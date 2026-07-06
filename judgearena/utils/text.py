"""Text helpers: truncation, safe coercion, and reasoning-tag stripping."""

from __future__ import annotations

import re

import pandas as pd

from judgearena.constants import VLLM_REASONING_END_STR


def truncate(s: str, max_len: int | None = None) -> str:
    """Truncate a string to *max_len* characters.

    Non-string inputs (e.g. ``None`` or ``float('nan')``) are coerced to the
    empty string so that callers don't have to guard against missing data.
    """
    if not isinstance(s, str):
        return ""
    if max_len is not None:
        return s[:max_len]
    return s


def safe_text(value: object, truncate_chars: int | None) -> str:
    """Coerce *value* to a string and optionally truncate.

    Returns the empty string for ``None`` and NaN-like values so callers
    don't have to guard against missing data.
    """
    if value is None:
        return ""
    is_missing = pd.isna(value)
    if isinstance(is_missing, bool) and is_missing:
        return ""
    return truncate(str(value), max_len=truncate_chars)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_thinking_tags(text: str | None) -> str:
    """Remove full `<think>...</think>` blocks from raw model output."""
    return strip_thinking_tags_with_metadata(text)[0]


def strip_thinking_tags_with_metadata(text: str | None) -> tuple[str, bool]:
    """Remove visible reasoning spans from raw model output."""
    if not isinstance(text, str):
        return "", False

    cleaned = _THINK_BLOCK_RE.sub("", text)
    if cleaned != text:
        return cleaned.lstrip(), True

    lowered = text.lower()
    closing_tag = "</think>"
    closing_idx = lowered.find(closing_tag)
    if closing_idx != -1 and "<think>" not in lowered[:closing_idx]:
        return text[closing_idx + len(closing_tag) :].lstrip(), True

    forced_end_idx = text.find(VLLM_REASONING_END_STR)
    if forced_end_idx != -1:
        return (
            text[forced_end_idx + len(VLLM_REASONING_END_STR) :].lstrip(),
            True,
        )

    return text, False
