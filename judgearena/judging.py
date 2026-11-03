"""Shared execution and parsing of prepared judge requests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, is_dataclass, replace

import pandas as pd
from langchain_core.prompt_values import PromptValue

from judgearena.battles import swap_comparison_slots
from judgearena.cache.inference import JudgementCacheRowMetadata
from judgearena.datasets.data import _require_columns, validate_messages
from judgearena.log import get_logger
from judgearena.models import InferenceResult, do_inference
from judgearena.prompts.parsing import (
    OutputParser,
    ParsedPreference,
    parser_name,
    validate_parser,
)
from judgearena.prompts.registry import ResolvedJudgePrompt


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


def execute_pairwise(
    model,
    comparisons: pd.DataFrame,
    render: Callable[[pd.DataFrame], Iterable[list[JudgeRequest[ParsedPreference]]]],
    *,
    swap_mode: str = "fixed",
    use_tqdm: bool = False,
) -> tuple[
    list[JudgeResult[ParsedPreference]], list[JudgeResult[ParsedPreference]] | None
]:
    """Execute rendered batches in order and restore comparison order per orientation.

    Renderers receive the actual A/B slots and return one request per row, using
    its stringified index as ID. Batches control scheduling, not result order.
    Model columns enable cache metadata; instruction_id/cache_orientation can
    retain a source battle's identity. Results keep preferences in judge slots.
    """
    if swap_mode not in ("fixed", "both"):
        raise ValueError("Pairwise execution expects fixed or both ordering.")
    ids = comparisons.index.map(str).tolist()
    if comparisons.index.isna().any() or len(set(ids)) != len(ids):
        raise ValueError("Comparisons require unique, nonmissing request IDs.")
    orders = []
    for swapped in (False, True) if swap_mode == "both" else (False,):
        frame = (
            swap_comparison_slots(comparisons, pd.Series(True, index=comparisons.index))
            if swapped
            else comparisons
        )
        batches = list(render(frame))
        request_ids = [request.id for batch in batches for request in batch]
        if len(request_ids) != len(ids) or set(request_ids) != set(ids):
            raise ValueError(
                "Rendered requests must cover each comparison exactly once."
            )
        metadata = {}
        if {"model_a", "model_b"} <= set(frame.columns):
            for row in frame.itertuples():
                orientation = getattr(row, "cache_orientation", "direct")
                metadata[str(row.Index)] = {
                    "instruction_id": str(getattr(row, "instruction_id", row.Index)),
                    "model_a": row.model_a,
                    "model_b": row.model_b,
                    "orientation": ("reversed" if orientation == "direct" else "direct")
                    if swapped
                    else orientation,
                }
        results = {}
        for batch in batches:
            requests = [
                replace(request, cache_metadata=metadata[request.id])
                if metadata
                else request
                for request in batch
            ]
            results.update(
                (result.request.id, result)
                for result in execute_judging(
                    model, requests, use_tqdm=use_tqdm, result_type=ParsedPreference
                )
            )
        orders.append([results[identity] for identity in ids])
    return orders[0], orders[1] if swap_mode == "both" else None


logger = get_logger(__name__)


def parsed_preferences(results) -> pd.Series:
    """Extract slot-oriented preferences, keeping failed parses as missing."""
    missing = sum(result.parsed is None for result in results)
    if missing:
        logger.warning(
            "%d/%d judge outputs could not be parsed; treating preferences as missing.",
            missing,
            len(results),
        )
    return pd.Series(
        [
            None if result.parsed is None else result.parsed.preference
            for result in results
        ],
        dtype=float,
    )


def annotation_frame(
    samples: pd.DataFrame,
    results: list[JudgeResult],
    prompts: Iterable[ResolvedJudgePrompt],
) -> pd.DataFrame:
    """Preserve source metadata and attach authoritative judge evidence in row order."""
    prompts = list(prompts)
    if len(results) != len(samples) or len(prompts) != len(results):
        raise ValueError("Each comparison requires one judge result and prompt.")
    return samples.reset_index(drop=True).assign(
        instruction_index=samples.index,
        judge_input=[
            result.request.input.to_string()
            if isinstance(result.request.input, PromptValue)
            else result.request.input
            for result in results
        ],
        judge_completion=[result.response.text for result in results],
        judge_top_logprobs=[
            result.response.first_token_top_logprobs
            if result.request.parser.requires_top_logprobs
            else None
            for result in results
        ],
        parsed=[
            asdict(result.parsed) if is_dataclass(result.parsed) else result.parsed
            for result in results
        ],
        prompt_preset=[prompt.preset_name for prompt in prompts],
        prompt_variant=[prompt.variant_name for prompt in prompts],
    )


def pairwise_annotation_frame(
    comparisons: pd.DataFrame,
    results: list[JudgeResult[ParsedPreference]],
    prompts: Iterable[ResolvedJudgePrompt],
    *,
    orientation: str,
    swapped: bool | list[bool] = False,
) -> pd.DataFrame:
    """Attach judge evidence to original comparison rows.

    Model/answer columns and pref use the original comparison slots. `swapped`
    records the actual input order; parsed scores still use the judge's slots.
    `orientation` identifies scoring repetitions (single/direct/reversed).
    Dataset metadata is retained, with result fields taking precedence.
    """
    frame = annotation_frame(comparisons, results, prompts).assign(
        swapped=swapped,
        orientation=orientation,
    )
    prefs = parsed_preferences(results)
    frame["pref"] = prefs.where(~frame["swapped"], 1 - prefs)
    return frame


def answer_samples(instructions: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    """Join canonical output rows by ID, keeping instruction and answer metadata.

    Output model/completion fields are authoritative. Instruction-side messages
    remain input_messages; output messages describe the completed conversation.
    Other overlapping metadata names must be disambiguated by the adapter.
    """
    _require_columns(instructions, {"instruction"}, "Answer samples")
    _require_columns(answers, {"model", "completion"}, "Answer outputs")
    if (
        instructions.index.isna().any()
        or not instructions.index.is_unique
        or not instructions.index.map(str).is_unique
        or not answers.index.is_unique
    ):
        raise ValueError("Samples and answers require unique, nonmissing request IDs.")
    context = instructions.drop(columns=["model", "completion"], errors="ignore")
    if "messages" in context:
        if "input_messages" in context:
            raise ValueError("Instruction messages overlap with input_messages.")
        context = context.rename(columns={"messages": "input_messages"})
    samples = context.join(answers)
    if samples[["instruction", "model", "completion"]].isna().any().any():
        raise ValueError(
            "Answer samples contain missing instructions, models or answers."
        )
    if "messages" in samples:
        for messages in samples["messages"].dropna():
            validate_messages(messages)
    return samples


def answer_text(completion, messages=None, *, strip_thinking=False) -> str:
    """Render a complete conversation when present, otherwise its text answer."""
    from judgearena.utils import strip_thinking_tags

    if isinstance(messages, list):
        return "\n".join(
            f"{message['role']}: "
            + (
                strip_thinking_tags(message["content"])
                if strip_thinking and message["role"] == "assistant"
                else message["content"]
            )
            for message in messages
        )
    return strip_thinking_tags(completion) if strip_thinking else completion


def judge_single_answers(
    model,
    samples: pd.DataFrame,
    render: Callable,
    *,
    use_tqdm: bool = False,
    result_type: type | None = None,
) -> tuple[pd.DataFrame, list[ResolvedJudgePrompt]]:
    """Grade indexed samples containing model/completion and renderer metadata.

    The renderer returns rendering-specific fields, parsed-result requests, and
    resolved prompts, all in sample order. Dataclasses and JSON-compatible results
    are preserved in ``parsed``; scalar result objects also populate ``score``.
    Requests use stringified row IDs;
    instruction_id overrides cache identity. Return annotations and their prompts.
    result_type restricts parsed outputs when a caller needs scalar scores.
    """
    ids = samples.index.map(str).tolist()
    if samples.index.isna().any() or len(set(ids)) != len(ids):
        raise ValueError("Samples require unique, nonmissing request IDs.")
    extras, requests, prompts = render(samples)
    for prompt in prompts:
        prompt.validate_for("single_answer", result_type)
    if (
        [request.id for request in requests] != ids
        or len(extras) != len(ids)
        or len(prompts) != len(ids)
    ):
        raise ValueError(
            "Score renderers must return fields, request, and prompt for each sample in order."
        )
    requests = [
        replace(
            request,
            cache_metadata={
                "instruction_id": str(getattr(row, "instruction_id", row.Index)),
                "model_a": row.model,
                "model_b": None,
            },
        )
        for request, row in zip(requests, samples.itertuples(), strict=True)
    ]
    results = execute_judging(
        model, requests, use_tqdm=use_tqdm, result_type=result_type
    )
    rendered = samples.assign(
        **{column: extras[column].to_numpy() for column in extras}
    )
    annotations = annotation_frame(rendered, results, prompts).assign(
        evaluated_model=samples["model"].to_numpy(),
    )
    annotations["score"] = annotations["parsed"].map(
        lambda parsed: parsed.get("score") if isinstance(parsed, dict) else None
    )
    return annotations, prompts
