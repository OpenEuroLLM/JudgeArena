"""Shared execution and parsing of prepared judge requests."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompt_values import PromptValue

from judgearena.cache.inference import JudgementCacheRowMetadata
from judgearena.log import get_logger
from judgearena.models import InferenceResult, do_inference
from judgearena.prompts.parsing import OutputParser, parser_name, validate_parser

logger = get_logger(__name__)


@dataclass(frozen=True)
class JudgeRequest[T]:
    id: str
    input: str | PromptValue
    parser: OutputParser[T]
    cache_metadata: JudgementCacheRowMetadata | None = None


@dataclass(frozen=True)
class JudgeResult[T]:
    request: JudgeRequest[T]
    response: InferenceResult
    parsed: T | None


def execute_judging[T](
    model,
    requests: list[JudgeRequest[T]],
    *,
    use_tqdm: bool = False,
    result_type: type[T] | None = None,
) -> list[JudgeResult[T]]:
    """Execute rendered requests and retain raw responses alongside parsed results."""
    if not requests:
        return []
    if len({request.id for request in requests}) != len(requests):
        raise ValueError("Judging request IDs must be unique within a batch.")
    for request in requests:
        validate_parser(request.parser, result_type)
    metadata = [request.cache_metadata for request in requests]
    if any(row is not None for row in metadata) and any(
        row is None for row in metadata
    ):
        raise ValueError("Cache metadata must be supplied for every request or none.")

    responses = do_inference(
        chat_model=model,
        inputs=[request.input for request in requests],
        use_tqdm=use_tqdm,
        return_top_logprobs=True,
        stage="judging",
        cache_row_metadata=metadata if metadata[0] is not None else None,
    )
    results = []
    missing_logprobs = 0
    for request, response in zip(requests, responses, strict=True):
        if (
            request.parser.requires_top_logprobs
            and not response.first_token_top_logprobs
        ):
            missing_logprobs += 1
            parsed = None
        else:
            parsed = request.parser.parse_result(
                response.text, top_logprobs=response.first_token_top_logprobs
            )
        if parsed is not None and not isinstance(parsed, request.parser.result_type):
            raise TypeError(
                f"Parser {parser_name(request.parser)!r} must return "
                f"{request.parser.result_type.__name__}; got {type(parsed).__name__}."
            )
        results.append(JudgeResult(request=request, response=response, parsed=parsed))
    if missing_logprobs:
        logger.warning(
            "%d judgments lacked required first-token logprobs; "
            "raw responses were retained and parsed results are missing.",
            missing_logprobs,
        )
    return results
