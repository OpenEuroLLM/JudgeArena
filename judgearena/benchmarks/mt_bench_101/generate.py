"""Golden-context completion generation for MT-Bench-101."""

from __future__ import annotations

from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from judgearena.models import do_inference, make_model
from judgearena.utils import truncate

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _escape_template_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _build_golden_context_input(
    *,
    system_prompt: str,
    golden_context: list[dict[str, str]],
    user_message: str,
    truncate_input_chars: int | None,
):
    messages: list[tuple[str, str]] = [
        ("system", _escape_template_braces(system_prompt))
    ]
    for turn in golden_context:
        messages.append(
            (
                "user",
                _escape_template_braces(
                    truncate(str(turn.get("user") or ""), max_len=truncate_input_chars)
                ),
            )
        )
        messages.append(
            (
                "assistant",
                _escape_template_braces(
                    truncate(str(turn.get("bot") or ""), max_len=truncate_input_chars)
                ),
            )
        )
    messages.append(
        (
            "user",
            _escape_template_braces(
                truncate(user_message, max_len=truncate_input_chars)
            ),
        )
    )
    return ChatPromptTemplate.from_messages(messages).invoke({})


def generate_mt_bench_101_completions(
    eval_items: pd.DataFrame,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 8192,
    use_tqdm: bool = True,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    **model_kwargs: Any,
) -> pd.DataFrame:
    """Generate MT-Bench-101 responses from golden-context eval items."""
    chat_model = make_model(model, max_tokens=max_tokens, **model_kwargs)
    inputs = [
        _build_golden_context_input(
            system_prompt=system_prompt,
            golden_context=list(row.get("golden_context") or []),
            user_message=str(row.get("user_message") or ""),
            truncate_input_chars=truncate_input_chars,
        )
        for _, row in eval_items.iterrows()
    ]
    completions = do_inference(
        chat_model=chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
        stage="generation",
    )
    idxs = list(eval_items.index)
    return pd.DataFrame(
        {
            "instruction_index": idxs,
            "dialogue_id": [eval_items.loc[idx, "dialogue_id"] for idx in idxs],
            "dialogue_uid": [eval_items.loc[idx, "dialogue_uid"] for idx in idxs],
            "task": [eval_items.loc[idx, "task"] for idx in idxs],
            "ability": [eval_items.loc[idx, "ability"] for idx in idxs],
            "turn_index": [eval_items.loc[idx, "turn_index"] for idx in idxs],
            "completion": completions,
        }
    )
