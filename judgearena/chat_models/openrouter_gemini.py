"""ChatOpenAI subclass tolerant to Gemini's PROHIBITED_CONTENT hard-refusals.

Google's core policy filter rejects a small fraction of prompts (e.g. graphic
violence, sexual content involving minors) with HTTP 403 ``PROHIBITED_CONTENT``
*regardless* of the adjustable ``safety_settings`` thresholds. These refusals
are legitimate, reproducible model behavior that a benchmark like
``m-arena-hard-v2.0`` surfaces: the baseline should contain them so the judge
can score them, not crash the run.

The subclass intercepts the error response before LangChain raises, returns
a stub assistant message with a clearly marked refusal payload and
``finish_reason="content_filter"``, and lets the rest of the pipeline proceed
unchanged.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

GEMINI_SAFETY_REFUSAL_MARKER = (
    "[Gemini safety refusal: PROHIBITED_CONTENT — Google's core policy filter "
    "blocked this prompt regardless of safety_settings.]"
)
OPENROUTER_GEMINI_SAFETY_REFUSAL_FINISH_REASON = "content_filter"

_PROHIBITED_CONTENT_TOKEN = "PROHIBITED_CONTENT"


def is_openrouter_gemini_model(model_spec: str) -> bool:
    """Return True when ``model_spec`` targets a Gemini model via OpenRouter.

    Matches ``OpenRouter/google/gemini-2.5-flash`` and related variants.
    """
    provider, sep, model_name = model_spec.partition("/")
    if not sep:
        return False
    lowered = model_name.lower()
    return provider == "OpenRouter" and (
        lowered.startswith("google/gemini") or lowered.startswith("google/gemma")
    )


def _error_is_prohibited_content(error: object) -> bool:
    if error is None:
        return False
    return _PROHIBITED_CONTENT_TOKEN in str(error)


def _build_prohibited_content_stub_payload(
    *, original_response: dict[str, Any], model_name: str
) -> dict[str, Any]:
    stub_message = {
        "role": "assistant",
        "content": GEMINI_SAFETY_REFUSAL_MARKER,
    }
    stub_choice = {
        "index": 0,
        "message": stub_message,
        "finish_reason": OPENROUTER_GEMINI_SAFETY_REFUSAL_FINISH_REASON,
    }
    return {
        "id": original_response.get("id") or "openrouter-gemini-safety-stub",
        "object": "chat.completion",
        "created": original_response.get("created") or 0,
        "model": original_response.get("model") or model_name,
        "choices": [stub_choice],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


class OpenRouterGeminiSafetyTolerantChatOpenAI(ChatOpenAI):
    """ChatOpenAI that converts Gemini PROHIBITED_CONTENT errors to stubs.

    Only intercepts the specific OpenRouter error surface for Gemini's core
    policy filter; all other errors propagate unchanged. The stub message has
    ``content == GEMINI_SAFETY_REFUSAL_MARKER`` and ``finish_reason ==
    "content_filter"`` so upstream validators and judges see the refusal
    explicitly rather than a silent drop.
    """

    def _create_chat_result(  # type: ignore[override]
        self,
        response,
        generation_info: dict | None = None,
    ):
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        error = response_dict.get("error")
        if _error_is_prohibited_content(error):
            stub = _build_prohibited_content_stub_payload(
                original_response=response_dict,
                model_name=self.model_name,
            )
            return super()._create_chat_result(stub, generation_info=generation_info)
        return super()._create_chat_result(response, generation_info=generation_info)
