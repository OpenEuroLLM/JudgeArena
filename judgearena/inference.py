"""Provider-neutral inference inputs and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langchain_core.prompt_values import PromptValue

from judgearena.usage import RequestUsage


class DecisionInput(TypedDict):
    """One state and its named questions; question schemas belong to the provider."""

    state: str | dict | list
    questions: dict[str, dict]


JudgeInput = str | PromptValue | DecisionInput


def validate_decision_input(value: DecisionInput) -> None:
    if not isinstance(value, dict) or set(value) != {"state", "questions"}:
        raise ValueError("Decision input requires state and questions.")
    if not isinstance(value["state"], (str, dict, list)):
        raise ValueError("Decision state must be text, an object, or an array.")
    questions = value["questions"]
    if (
        not isinstance(questions, dict)
        or not questions
        or not all(
            isinstance(name, str) and name and isinstance(question, dict)
            for name, question in questions.items()
        )
    ):
        raise ValueError(
            "Decision questions must be a nonempty mapping of names to objects."
        )


@dataclass(frozen=True)
class InferenceResult:
    """A text completion and optional provider response details."""

    text: str
    first_token_top_logprobs: dict[str, float] | None = None
    usage: RequestUsage | None = None
