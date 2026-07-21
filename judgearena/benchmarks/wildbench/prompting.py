"""Official WildBench V2 prompt templates and renderers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

import pandas as pd

from judgearena.utils import truncate


@dataclass(frozen=True)
class WildBenchPrompt:
    """Registered WildBench prompt resource and supported judging mode."""

    name: str
    mode: Literal["score", "reward"]
    resource: str

    @property
    def template(self) -> str:
        return (
            files("judgearena.prompts")
            .joinpath(self.resource)
            .read_text(encoding="utf-8")
        )


_WILDBENCH_PROMPTS = {
    "wildbench-score-v2": WildBenchPrompt(
        name="wildbench-score-v2",
        mode="score",
        resource="wildbench/score-v2.txt",
    ),
    "wildbench-pairwise-v2": WildBenchPrompt(
        name="wildbench-pairwise-v2",
        mode="reward",
        resource="wildbench/pairwise-v2.txt",
    ),
}

WILDBENCH_PROMPT_NAMES = frozenset(_WILDBENCH_PROMPTS)


def resolve_wildbench_prompt(
    name: str, *, mode: Literal["score", "reward"]
) -> WildBenchPrompt:
    """Return a registered prompt and verify its judging mode."""
    try:
        prompt = _WILDBENCH_PROMPTS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown WildBench prompt {name!r}; available: "
            f"{sorted(WILDBENCH_PROMPT_NAMES)}"
        ) from exc
    if prompt.mode != mode:
        raise ValueError(
            f"WildBench {mode} mode cannot use {prompt.mode} prompt {name!r}."
        )
    return prompt


def _shorten_words(text: object, max_words: int) -> str:
    value = "" if text is None else str(text)
    words = value.split(" ")
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "... (truncated)"
    return value


def _prompt_field(text: object, *, max_words: int, max_chars: int | None) -> str:
    return truncate(_shorten_words(text, max_words), max_len=max_chars)


def _checklist_markdown(checklist: object) -> str:
    if not isinstance(checklist, list):
        return ""
    return "".join(f"- {item}\n" for item in checklist)


def render_wildbench_score_prompt(
    prompt: WildBenchPrompt,
    example: pd.Series,
    model_output: str,
    *,
    max_words: int,
    max_chars: int | None,
) -> str:
    """Render the official WB-Score prompt for one example."""
    replacements = {
        "$HISTORY": _prompt_field(
            example["history"], max_words=max_words, max_chars=max_chars
        ),
        "$USER_QUERY": _prompt_field(
            example["instruction"], max_words=max_words, max_chars=max_chars
        ),
        "$MODEL_OUTPUT": _prompt_field(
            model_output, max_words=max_words, max_chars=max_chars
        ),
        "$CHECKLIST": _checklist_markdown(example["checklist"]),
    }
    rendered = prompt.template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def render_wildbench_pairwise_prompt(
    prompt: WildBenchPrompt,
    example: pd.Series,
    completion_a: str,
    completion_b: str,
    *,
    max_words: int,
    max_chars: int | None,
) -> str:
    """Render the official WB-Reward pairwise prompt for one example."""
    replacements = {
        "$HISTORY": _prompt_field(
            example["history"], max_words=max_words, max_chars=max_chars
        ),
        "$USER_QUERY": _prompt_field(
            example["instruction"], max_words=max_words, max_chars=max_chars
        ),
        "$CANDIDATE_A": _prompt_field(
            completion_a or "[This model response is empty.]",
            max_words=max_words,
            max_chars=max_chars,
        ),
        "$CANDIDATE_B": _prompt_field(
            completion_b or "[This model response is empty.]",
            max_words=max_words,
            max_chars=max_chars,
        ),
        "$CHECKLIST": _checklist_markdown(example["checklist"]),
    }
    rendered = prompt.template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered
