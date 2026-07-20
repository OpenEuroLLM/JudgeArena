"""Shared helpers for cache backfill extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

HOSTED_BACKFILL_PROVIDERS = frozenset(
    {"OpenRouter", "ChatOpenAI", "OpenAI", "Together", "Dummy"}
)


def provider_from_model_spec(model_spec: str) -> str:
    provider, _, _ = model_spec.partition("/")
    return provider


def is_backfillable_provider(model_spec: str) -> bool:
    return provider_from_model_spec(model_spec) in HOSTED_BACKFILL_PROVIDERS


def source_run_id(run_dir: Path) -> str:
    return run_dir.name


def chat_prompt_value(
    *, system_prompt: str | None, user_prompt: str
) -> ChatPromptValue:
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=user_prompt))
    return ChatPromptValue(messages=messages)


def prompt_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    if not text.strip() or text.strip().lower() == "nan":
        return None
    return text


def mt_swapped(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def increment(skipped: dict[str, int], reason: str, count: int = 1) -> None:
    skipped[reason] = skipped.get(reason, 0) + count
