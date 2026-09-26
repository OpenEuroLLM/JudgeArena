"""Provider-neutral inference result types."""

from __future__ import annotations

from dataclasses import dataclass

from judgearena.usage import RequestUsage


@dataclass(frozen=True)
class InferenceResult:
    """A text completion and optional provider response details."""

    text: str
    first_token_top_logprobs: dict[str, float] | None = None
    usage: RequestUsage | None = None
