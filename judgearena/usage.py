"""Model request usage collected during one benchmark run."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestUsage:
    """Usage record for one model request."""

    stage: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def has_token_usage(self) -> bool:
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.total_tokens,
                self.reasoning_tokens,
                self.cached_tokens,
            )
        )


def _sum_optional(requests: tuple[RequestUsage, ...], field: str) -> int | None:
    values = [getattr(request, field) for request in requests]
    reported = [int(value) for value in values if value is not None]
    return sum(reported) if reported else None


def _summarize(requests: tuple[RequestUsage, ...]) -> dict[str, object]:
    token_fields = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reasoning_tokens",
        "cached_tokens",
    )
    reported = {
        field: sum(getattr(request, field) is not None for request in requests)
        for field in token_fields
    }
    cost_values = [
        request.cost_usd for request in requests if request.cost_usd is not None
    ]
    return {
        "requests": len(requests),
        **{field: _sum_optional(requests, field) for field in token_fields},
        "cost_usd": sum(cost_values) if cost_values else None,
        "requests_with_any_token_usage": sum(
            request.has_token_usage for request in requests
        ),
        **{f"requests_with_{field}": count for field, count in reported.items()},
        "requests_with_cost": len(cost_values),
    }


@dataclass(frozen=True)
class RunUsage:
    """Immutable snapshot of the model requests made during a run."""

    requests: tuple[RequestUsage, ...]

    def summary(self) -> dict[str, object]:
        return _summarize(self.requests)

    def format_summary(self) -> str:
        return _format_summary(self.summary())

    def to_dict(self) -> dict[str, object]:
        by_stage = {
            stage: _summarize(
                tuple(request for request in self.requests if request.stage == stage)
            )
            for stage in sorted({request.stage for request in self.requests})
        }
        by_model = {
            model: _summarize(
                tuple(
                    request
                    for request in self.requests
                    if (request.model or "unknown") == model
                )
            )
            for model in sorted(
                {request.model or "unknown" for request in self.requests}
            )
        }
        return {
            "scope": "successful_model_responses_returned_by_completed_inference_calls",
            "source": "provider_response_metadata_when_available",
            "total": self.summary(),
            "by_stage": by_stage,
            "by_model": by_model,
        }

    def render(self) -> None:
        data = self.to_dict()
        total = data["total"]
        assert isinstance(total, dict)

        print("\nModel usage:")
        if not self.requests:
            print("  No model requests were made during this run.")
            return

        by_stage = data["by_stage"]
        assert isinstance(by_stage, dict)
        for stage, summary in by_stage.items():
            assert isinstance(summary, dict)
            print(f"  {stage.capitalize()}: {_format_summary(summary)}")
        print(f"  Total: {self.format_summary()}")


def _format_summary(summary: dict[str, object]) -> str:
    requests = int(summary["requests"])
    input_tokens = summary["input_tokens"]
    output_tokens = summary["output_tokens"]
    cost = summary["cost_usd"]
    input_reported = int(summary["requests_with_input_tokens"])
    output_reported = int(summary["requests_with_output_tokens"])
    cost_reported = int(summary["requests_with_cost"])

    if input_tokens is None and output_tokens is None:
        tokens = "token usage unavailable"
    else:
        input_text = f"{int(input_tokens):,}" if input_tokens is not None else "unknown"
        output_text = (
            f"{int(output_tokens):,}" if output_tokens is not None else "unknown"
        )
        tokens = f"{input_text} input / {output_text} output tokens"
        if input_reported < requests or output_reported < requests:
            tokens += (
                " reported (partial: "
                f"input {input_reported}/{requests}, "
                f"output {output_reported}/{requests})"
            )

    if requests == 0:
        cost_text = "$0.000000"
    elif cost_reported == 0:
        cost_text = "cost unavailable"
    elif cost_reported < requests:
        cost_text = f"${float(cost):.6f} reported (partial)"
    else:
        cost_text = f"${float(cost):.6f}"
    return f"{requests} successful response(s), {tokens}, {cost_text}"


class UsageTracker:
    """Collect request usage without changing benchmark runner signatures."""

    def __init__(self) -> None:
        self._requests: list[RequestUsage] = []

    def record(self, usage: RequestUsage) -> None:
        self._requests.append(usage)

    def record_many(self, usage: list[RequestUsage]) -> None:
        self._requests.extend(usage)

    def snapshot(self) -> RunUsage:
        return RunUsage(requests=tuple(self._requests))

    def render_summary(self) -> None:
        self.snapshot().render()


_CURRENT_TRACKER: ContextVar[UsageTracker | None] = ContextVar(
    "judgearena_usage_tracker", default=None
)


@contextmanager
def track_usage() -> Iterator[UsageTracker]:
    """Collect usage; nested scopes intentionally share the outer run tracker."""
    current = _CURRENT_TRACKER.get()
    if current is not None:
        yield current
        return

    tracker = UsageTracker()
    token = _CURRENT_TRACKER.set(tracker)
    try:
        yield tracker
    finally:
        _CURRENT_TRACKER.reset(token)


def record_usage(requests: list[RequestUsage]) -> None:
    tracker = _CURRENT_TRACKER.get()
    if tracker is not None:
        tracker.record_many(requests)


def current_run_usage() -> RunUsage | None:
    tracker = _CURRENT_TRACKER.get()
    return tracker.snapshot() if tracker is not None else None
