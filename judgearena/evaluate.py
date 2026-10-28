from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import partial

import pandas as pd
from langchain_core.prompt_values import PromptValue

from judgearena.cache.inference import JudgementCacheRowMetadata
from judgearena.judging import (
    JudgeRequest,
    answer_text,
    execute_judging,
    judge_single_answers,
)
from judgearena.log import get_logger
from judgearena.models import InferenceResult, do_inference
from judgearena.prompts.registry import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    ResolvedJudgePrompt,
    build_chat_prompt_template,
    group_judge_prompts,
    resolve_judge_prompt,
)
from judgearena.prompts.registry import (
    resolve_run_judge_prompt as resolve_run_judge_prompt,
)
from judgearena.utils import strip_thinking_tags, truncate

logger = get_logger(__name__)

# Re-exported for existing callers; implementations live with the presets.
from judgearena.prompts.parsing import (  # noqa: E402
    JudgeParser,
    OutputParser,
    PairScore,  # noqa: F401
    ParsedPreference,
    resolve_judge_parser,
)


def resolve_judge_prompts(
    *,
    multi_turn: bool = False,
    prompt_preset: str | None = None,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    task: str | None = None,
    system_file: str | None = None,
    user_file: str | None = None,
    parser: str | None = None,
) -> ResolvedJudgePrompt:
    if system_prompt is not None and user_prompt_template is not None:
        return ResolvedJudgePrompt(
            preset_name=prompt_preset or "custom",
            parser=resolve_judge_parser(parser or "score"),
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            source="override",
        )
    if system_prompt is not None or user_prompt_template is not None:
        raise ValueError(
            "Both system_prompt and user_prompt_template must be provided together."
        )

    resolved = resolve_judge_prompt(
        task=task,
        preset=prompt_preset,
        system_file=system_file,
        user_file=user_file,
        multi_turn=multi_turn,
        parser=parser,
    )
    if resolved.delegated:
        raise ValueError(
            f"Judge prompt preset '{resolved.preset_name}' is delegated and cannot "
            "be used for generic pairwise judging."
        )
    return resolved


@dataclass
class JudgeAnnotation:
    instruction: str  # instruction from the user
    completion_A: str  # completion of the first model
    completion_B: str  # completion of the second model
    judge_completion: str  # output of the judge
    judge_input: str | None = None  # input that was passed to the judge
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET
    # first-token top logprobs, only collected for logprob-weighted presets
    judge_top_logprobs: dict[str, float] | None = None
    parsed: ParsedPreference | None = None


def _prompt_fields(**fields):
    return {
        **fields,
        **{
            f"{key}_json": json.dumps(value, ensure_ascii=False)
            for key, value in fields.items()
        },
    }


def validate_prompt_fields(prompt: ResolvedJudgePrompt, columns, supplied) -> None:
    """Require template fields on the prepared rows used for judging."""
    required = set(build_chat_prompt_template(prompt).input_variables)
    if columns is None:
        return  # Lazy output metadata is checked once its rows are loaded.
    missing = (
        required - set(columns) - _prompt_fields(**dict.fromkeys(supplied, "")).keys()
    )
    if missing:
        raise ValueError(f"Judge prompt requires dataset columns: {sorted(missing)}.")


def validate_battle_prompt_fields(prompt: ResolvedJudgePrompt, columns) -> None:
    validate_prompt_fields(
        prompt,
        columns,
        (
            "user_prompt",
            "completion_A",
            "completion_B",
            "completion_A_text",
            "completion_B_text",
        ),
    )


def render_battle_inputs(
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    prompt: ResolvedJudgePrompt,
    truncate_input_chars: int | None,
    *,
    metadata: list[dict[str, object]] | None = None,
) -> list[PromptValue]:
    """Render text comparisons, preserving JSON escaping and answer truncation."""
    prompt_template = build_chat_prompt_template(prompt)

    prompt_inputs = []
    for user_prompt, completion_A, completion_B, row in zip(
        instructions,
        completions_A,
        completions_B,
        metadata if metadata is not None else [{}] * len(instructions),
        strict=True,
    ):
        completion_A = truncate(completion_A, max_len=truncate_input_chars)
        completion_B = truncate(completion_B, max_len=truncate_input_chars)
        prompt_inputs.append(
            {
                **row,
                **_prompt_fields(
                    user_prompt=user_prompt,
                    completion_A=completion_A,
                    completion_B=completion_B,
                    completion_A_text=truncate(
                        row.get("completion_a_text", completion_A),
                        max_len=truncate_input_chars,
                    ),
                    completion_B_text=truncate(
                        row.get("completion_b_text", completion_B),
                        max_len=truncate_input_chars,
                    ),
                ),
            }
        )
    return prompt_template.batch(prompt_inputs)


def annotate_battles(
    judge_chat_model,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    system_prompt: str | None = None,
    user_prompt_template: str = None,
    truncate_input_chars: int | None = 8192,
    use_tqdm: bool = False,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    strip_thinking_before_judging: bool = False,
    collect_top_logprobs: bool = False,
    cache_row_metadata: list[JudgementCacheRowMetadata] | None = None,
    *,
    parse: OutputParser[ParsedPreference] | None = None,
    prompt: ResolvedJudgePrompt | None = None,
) -> list[JudgeAnnotation]:
    """
    Directly evaluate from list of instructions and completions
    Can also pass custom LLM judge prompts, if not passed uses defaults
    Example usage:
    ```python
    annotations = annotate_battles(
        # can be any langchain ChatModel, supports OpenAI, Together, vLLM, ...
        judge_chat_model=Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        # the instructions we want to evaluate
        instructions=["Write numbers between 1 and 5."],
        # the completions we want to evaluate for the first model
        completions_A=["1 2 3 4 5."],
        # the completions we want to evaluate for the second model
        completions_B=["No"],
    )
    ```
    :param judge_chat_model:
    :param instructions:
    :param completions_A:
    :param completions_B:
    :param system_prompt:
    :param user_prompt_template:
    :param truncate_input_chars: Max characters to truncate completions before sending to judge.
    :param use_tqdm:
    :return:
    """
    # alternatively pass list of tuples
    assert len(instructions) == len(completions_A) == len(completions_B)

    resolved_prompt = prompt
    if resolved_prompt is None:
        resolved_prompt = resolve_judge_prompts(
            prompt_preset=prompt_preset,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )
    else:
        parse = parse if parse is not None else resolved_prompt.parser
        if parse is not None:
            collect_top_logprobs |= parse.requires_top_logprobs
    if resolved_prompt.delegated:
        raise ValueError(
            f"Judge prompt preset '{resolved_prompt.preset_name}' is delegated and "
            "cannot be used for generic pairwise judging."
        )
    effective_prompt = (
        replace(resolved_prompt, parser=parse) if parse is not None else resolved_prompt
    )
    effective_prompt.validate_for("pairwise")

    if strip_thinking_before_judging:
        completions_A = [strip_thinking_tags(c) for c in completions_A]
        completions_B = [strip_thinking_tags(c) for c in completions_B]
    inputs = render_battle_inputs(
        instructions,
        completions_A,
        completions_B,
        resolved_prompt,
        truncate_input_chars,
    )

    logger.info("Start LLM judge annotation (%d annotations).", len(inputs))
    if parse is None:
        judge_results = do_inference(
            chat_model=judge_chat_model,
            inputs=inputs,
            use_tqdm=use_tqdm,
            return_top_logprobs=collect_top_logprobs,
            stage="judging",
            cache_row_metadata=cache_row_metadata,
        )
        if not collect_top_logprobs:
            judge_results = [InferenceResult(text=text) for text in judge_results]
        outcomes = ((response, None) for response in judge_results)
    else:
        metadata = (
            cache_row_metadata
            if cache_row_metadata is not None
            else [None] * len(inputs)
        )
        requests = [
            JudgeRequest(
                id=str(index),
                input=judge_input,
                parser=parse,
                cache_metadata=row,
            )
            for index, (judge_input, row) in enumerate(
                zip(inputs, metadata, strict=True)
            )
        ]
        results = execute_judging(
            judge_chat_model, requests, use_tqdm=use_tqdm, result_type=ParsedPreference
        )
        outcomes = ((result.response, result.parsed) for result in results)

    return [
        JudgeAnnotation(
            judge_input=judge_input,
            judge_completion=response.text,
            instruction=instruction,
            completion_A=completion_A,
            completion_B=completion_B,
            prompt_preset=resolved_prompt.preset_name,
            judge_top_logprobs=(
                response.first_token_top_logprobs if collect_top_logprobs else None
            ),
            parsed=parsed,
        )
        for (
            judge_input,
            (response, parsed),
            instruction,
            completion_A,
            completion_B,
        ) in zip(
            inputs,
            outcomes,
            instructions,
            completions_A,
            completions_B,
            strict=True,
        )
    ]


def render_comparison_requests(comparisons, prompt, truncate_input_chars):
    """Render a text batch for the shared executor."""
    if prompt.parser is None or prompt.delegated:
        raise ValueError("Pairwise judging requires a resolved preference parser.")
    inputs = render_battle_inputs(
        comparisons.instruction.tolist(),
        comparisons.completion_a.tolist(),
        comparisons.completion_b.tolist(),
        prompt,
        truncate_input_chars,
        metadata=comparisons.to_dict("records"),
    )
    return [
        [
            JudgeRequest(str(identity), value, prompt.parser)
            for identity, value in zip(comparisons.index, inputs, strict=True)
        ]
    ]


def render_single_requests(
    samples, *, prompt, truncate_input_chars, strip_thinking_before_judging
):
    """Render one answer with its metadata; preserve parser-defined result types."""
    validate_prompt_fields(
        prompt, samples.columns, ("user_prompt", "completion", "completion_text")
    )
    inputs = []
    for row in samples.to_dict("records"):
        completion = answer_text(
            row["completion"],
            row.get("messages"),
            strip_thinking=strip_thinking_before_judging,
        )
        inputs.append(
            {
                **row,
                **_prompt_fields(
                    user_prompt=row["instruction"],
                    completion=truncate(completion, max_len=truncate_input_chars),
                    completion_text=truncate(
                        answer_text(
                            row["completion"],
                            strip_thinking=strip_thinking_before_judging,
                        ),
                        max_len=truncate_input_chars,
                    ),
                ),
            }
        )
    rendered = build_chat_prompt_template(prompt).batch(inputs)
    requests = [
        JudgeRequest(str(identity), value, prompt.parser)
        for identity, value in zip(samples.index, rendered, strict=True)
    ]
    return samples.iloc[:, :0], requests, [prompt] * len(samples)


def prepare_single_answer_judging(
    cfg,
    resolved_prompt,
    instructions,
    *,
    result_type: type | None = None,
    use_tqdm: bool | None = None,
    available_columns=None,
):
    """Validate prompts against known prepared fields before generating answers."""
    use_tqdm = cfg.run.use_tqdm if use_tqdm is None else use_tqdm
    groups = group_judge_prompts(cfg, instructions, resolved_prompt)
    renderers = []
    for prompt, ids in groups:
        prompt.validate_for(
            "single_answer", result_type, inference_kwargs=cfg.judge_model_kwargs()
        )
        kwargs = dict(
            truncate_input_chars=cfg.generation.truncate_judge_input_chars,
            strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        )
        if prompt.score_renderer is not None:
            render = partial(
                prompt.score_renderer, prompt_preset=prompt.preset_name, **kwargs
            )
        else:
            if prompt.delegated:
                raise ValueError(
                    "Independent judging requires a parser and a renderable prompt."
                )
            validate_prompt_fields(
                prompt,
                None
                if available_columns is None
                else set(available_columns) | {"instruction", "model", "completion"},
                ("user_prompt", "completion", "completion_text"),
            )
            render = partial(render_single_requests, prompt=prompt, **kwargs)
        renderers.append((ids, render))

    def judge(model, samples, *, source_ids=None):
        frames, prompts = [], []
        routing = samples.index if source_ids is None else source_ids.loc[samples.index]
        for ids, render in renderers:
            selected = samples.loc[routing.isin(ids)]
            if not selected.empty:
                frame, used = judge_single_answers(
                    model,
                    selected,
                    render,
                    use_tqdm=use_tqdm,
                    result_type=result_type,
                )
                frames.append(frame)
                prompts.extend(used)
        annotations = (
            pd.concat(frames, ignore_index=True)
            .set_index("instruction_index", drop=False)
            .loc[samples.index]
            .reset_index(drop=True)
        )
        return annotations, prompts

    return judge


def combine_swapped_prefs(prefs_ab: pd.Series, prefs_ba: pd.Series) -> pd.Series:
    """Stack direct preferences before reversed preferences reoriented to A/B."""
    return pd.concat(
        [prefs_ab.reset_index(drop=True), 1 - prefs_ba.reset_index(drop=True)]
    ).reset_index(drop=True)


def judge_and_parse_prefs(
    judge_chat_model,
    instructions: list[str],
    completions_A: list[str],
    completions_B: list[str],
    swap_mode: str = "fixed",
    strip_thinking_before_judging: bool = False,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    truncate_input_chars: int = 8192,
    use_tqdm: bool = False,
    parse: JudgeParser | None = None,
    cache_row_metadata: list[JudgementCacheRowMetadata] | None = None,
) -> tuple[list[JudgeAnnotation], list[JudgeAnnotation] | None, pd.Series]:
    """Run judge annotation and parse preferences, handling swap_mode='both'.

    Returns:
        annotations: original-order JudgeAnnotations
        annotations_reversed: reversed-order JudgeAnnotations (None if swap_mode != "both")
        prefs: canonical A/B preferences. With swap_mode="both", the direct
               block is followed by the reoriented reversed block.
    """
    if parse is None:
        parse = resolve_judge_prompts(
            prompt_preset=prompt_preset,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        ).parser

    if swap_mode == "both":
        logger.info(
            "Correction for judge bias towards a certain model position is set."
        )
        logger.info(
            "Evaluating completions with models reversed with judge %s.",
            judge_chat_model,
        )

    annotations = annotate_battles(
        judge_chat_model=judge_chat_model,
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        strip_thinking_before_judging=strip_thinking_before_judging,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        prompt_preset=prompt_preset,
        truncate_input_chars=truncate_input_chars,
        use_tqdm=use_tqdm,
        collect_top_logprobs=parse.requires_top_logprobs,
        parse=parse,
        cache_row_metadata=cache_row_metadata,
    )

    annotations_reversed = None
    if swap_mode == "both":
        reversed_cache_row_metadata = (
            [
                {
                    **metadata,
                    "model_a": metadata["model_b"],
                    "model_b": metadata["model_a"],
                    "orientation": (
                        "reversed"
                        if metadata.get("orientation") == "direct"
                        else "direct"
                    ),
                }
                for metadata in cache_row_metadata
            ]
            if cache_row_metadata is not None
            else None
        )
        annotations_reversed = annotate_battles(
            judge_chat_model=judge_chat_model,
            instructions=instructions,
            completions_A=completions_B,
            completions_B=completions_A,
            strip_thinking_before_judging=strip_thinking_before_judging,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            prompt_preset=prompt_preset,
            truncate_input_chars=truncate_input_chars,
            use_tqdm=use_tqdm,
            collect_top_logprobs=parse.requires_top_logprobs,
            parse=parse,
            cache_row_metadata=reversed_cache_row_metadata,
        )

    def _none_to_nan(x):
        return float("nan") if x is None else x

    def _parse_and_warn(ann_list: list, label: str) -> pd.Series:
        results = [annotation.parsed for annotation in ann_list]
        n_failed = sum(1 for result in results if result is None)
        if n_failed:
            logger.warning(
                "%d/%d judge outputs could not be parsed (%s) — "
                "those outputs are treated as missing preferences.",
                n_failed,
                len(results),
                label,
            )
        return pd.Series(
            [None if result is None else result.preference for result in results]
        )

    prefs = _parse_and_warn(annotations, "direct")

    if swap_mode == "both":
        prefs_reversed = _parse_and_warn(annotations_reversed, "reversed").apply(
            _none_to_nan
        )
        prefs = combine_swapped_prefs(prefs.apply(_none_to_nan), prefs_reversed)

    return annotations, annotations_reversed, prefs
