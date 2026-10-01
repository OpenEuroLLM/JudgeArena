"""OpenRouter System One backend for Jev judges."""

from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from judgearena.inference import InferenceResult
from judgearena.log import get_logger
from judgearena.prompts.jev import JEV_QUESTION_MODES
from judgearena.retry import _is_retryable_error
from judgearena.usage import RequestUsage

logger = get_logger(__name__)

OPENROUTER_JEV_ENDPOINT = "https://openrouter.ai/api/v1/systemone"


def _is_retryable_jev_error(error: Exception) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in {
            408,
            429,
            500,
            502,
            503,
            504,
            520,
            524,
            529,
        }
    return _is_retryable_error(error)


class OpenRouterJevJudge:
    """Pairwise judge using Jev through OpenRouter's System One API."""

    endpoint = OPENROUTER_JEV_ENDPOINT

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 30,
        max_concurrency: int = 16,
        decision_mode: str = "choice",
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ):
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key and (client is None or async_client is None):
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter Jev.")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive.")
        if decision_mode not in JEV_QUESTION_MODES:
            raise ValueError(
                f"Unknown Jev decision_mode {decision_mode!r}; expected one of "
                f"{sorted(JEV_QUESTION_MODES)}."
            )
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.max_concurrency = max_concurrency
        self.decision_mode = decision_mode
        self.client = client or httpx.Client(headers=headers, timeout=timeout)
        self.async_client = async_client or httpx.AsyncClient(
            headers=headers, timeout=timeout
        )
        self.questions = JEV_QUESTION_MODES[decision_mode]

    @staticmethod
    def _state(input_item):
        messages = getattr(input_item, "messages", None)
        if not messages:
            return {"comparison": str(input_item)}

        instructions = "\n\n".join(
            str(message.content)
            for message in messages
            if getattr(message, "type", None) == "system"
        )
        comparison_text = str(messages[-1].content)
        try:
            comparison = json.loads(comparison_text)
        except json.JSONDecodeError:
            comparison = comparison_text
        return {
            "evaluation_instructions": instructions,
            "comparison": comparison,
        }

    def _request(self, input_item) -> dict[str, object]:
        return {
            "model": self.model,
            "state": self._state(input_item),
            "questions": self.questions,
        }

    def _result(self, response: dict, stage: str) -> InferenceResult:
        answers = response["answers"]
        if self.decision_mode == "choice":
            answer = answers["preference"]
            if answer.get("type") != "choice" or "probabilities" not in answer:
                raise ValueError(
                    "OpenRouter Jev preference answer must include Choice "
                    "probabilities."
                )
            result_payload = dict(answer)
        elif self.decision_mode == "comparative-score":
            answer = answers.get("outcome", {})
            if (
                set(answers) != {"outcome"}
                or answer.get("type") != "score"
                or "probabilities" not in answer
            ):
                raise ValueError(
                    "OpenRouter Jev overall comparative answer must include one "
                    "Score distribution."
                )
            result_payload = {
                "type": "overall_comparative_score",
                "decision_mode": self.decision_mode,
                "answers": {"outcome": answer},
            }
        elif self.decision_mode == "multilingual-choice":
            answer = answers.get("outcome", {})
            if (
                set(answers) != {"outcome"}
                or answer.get("type") != "choice"
                or "probabilities" not in answer
            ):
                raise ValueError(
                    "OpenRouter Jev overall-choice answer must include one Choice "
                    "distribution."
                )
            result_payload = {
                "type": "overall_choice",
                "decision_mode": self.decision_mode,
                "answers": {"outcome": answer},
            }
        else:
            raise ValueError(f"Unsupported Jev decision mode: {self.decision_mode}.")
        usage = response["usage"]
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = (
            input_tokens + output_tokens
            if input_tokens is not None and output_tokens is not None
            else None
        )
        return InferenceResult(
            text=json.dumps(
                {
                    **result_payload,
                    "model": response["model"],
                    "request_id": response.get("id"),
                },
                sort_keys=True,
            ),
            usage=RequestUsage(
                stage=stage,
                model=f"OpenRouter/{response['model']}",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=usage.get("cost"),
            ),
        )

    def _post_with_retry(self, payload: dict[str, object]) -> dict:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self.client.post(self.endpoint, json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                if not _is_retryable_jev_error(exc):
                    raise
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"OpenRouter Jev request failed after {max_retries} attempts."
                    ) from exc
                delay = 2**attempt
                logger.warning(
                    "Retrying OpenRouter Jev request after %s (%d/%d) in %ss.",
                    type(exc).__name__,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    async def _apost_with_retry(self, payload: dict[str, object]) -> dict:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await self.async_client.post(self.endpoint, json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                if not _is_retryable_jev_error(exc):
                    raise
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"OpenRouter Jev request failed after {max_retries} attempts."
                    ) from exc
                delay = 2**attempt
                logger.warning(
                    "Retrying OpenRouter Jev request after %s (%d/%d) in %ss.",
                    type(exc).__name__,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    def invoke(self, input_item, *, usage_stage: str = "judging", **_kwargs):
        return self._result(
            self._post_with_retry(self._request(input_item)), usage_stage
        )

    def batch(self, inputs, *, usage_stage: str = "judging", **kwargs):
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            return list(
                executor.map(
                    lambda input_item: self.invoke(
                        input_item, usage_stage=usage_stage, **kwargs
                    ),
                    inputs,
                )
            )

    async def ainvoke(self, input_item, *, usage_stage: str = "judging", **_kwargs):
        response = await self._apost_with_retry(self._request(input_item))
        return self._result(response, usage_stage)
