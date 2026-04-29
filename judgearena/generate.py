from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from judgearena.utils import (
    LimitEventTracker,
    do_inference,
    make_model,
    strip_thinking_tags_with_metadata,
    truncate_with_metadata,
)


def _record_generation_output_limit_events(
    *,
    metadata: list[dict[str, Any]],
    case_ids: list[object],
    field: str,
    model_spec: str,
    limit_event_tracker: LimitEventTracker | None,
) -> list[bool]:
    hit_token_limit: list[bool] = []
    for case_id, metadata_row in zip(case_ids, metadata, strict=True):
        row = metadata_row or {}
        finish_reason = str(row.get("finish_reason") or "").lower()
        reached_limit = finish_reason == "length"
        hit_token_limit.append(reached_limit)
        if limit_event_tracker is None:
            continue
        if reached_limit:
            limit_event_tracker.record(
                "generation_output_token_limit",
                stage="generation_output",
                field=field,
                case_id=case_id,
                model_spec=model_spec,
                note=finish_reason,
            )
        if row.get("thinking_budget_exhausted"):
            limit_event_tracker.record(
                "generation_thinking_token_budget",
                stage="generation_output",
                field=field,
                case_id=case_id,
                model_spec=model_spec,
                note=str(row.get("thinking_token_budget")),
            )
    return hit_token_limit


def generate_instructions(
    instructions: pd.Series,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 32768,
    use_tqdm: bool = True,
    system_prompt: str | None = None,
    usage_tracker=None,
    usage_phase: str | None = None,
    limit_event_tracker: LimitEventTracker | None = None,
    **engine_kwargs,
) -> pd.DataFrame:
    chat_model = make_model(
        model,
        max_tokens=max_tokens,
        limit_event_tracker=limit_event_tracker,
        limit_event_stage="generation_model_init",
        limit_event_model_spec=model,
        **engine_kwargs,
    )

    # TODO improve prompt to generate instructions
    if system_prompt is None:
        system_prompt = (
            "You are an helpful assistant that answer queries asked by users."
        )
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{user_prompt}")]
    )

    prompt_truncated: list[bool] = []
    input_payloads = []
    case_ids = instructions.index.tolist()
    for case_id, user_prompt in zip(case_ids, instructions, strict=True):
        truncated_user_prompt, was_truncated = truncate_with_metadata(
            user_prompt,
            max_len=truncate_input_chars,
            tracker=limit_event_tracker,
            kind="generation_input_char_truncation",
            stage="generation_input",
            field="user_prompt",
            case_id=case_id,
            model_spec=model,
        )
        prompt_truncated.append(was_truncated)
        input_payloads.append({"user_prompt": truncated_user_prompt})
    inputs = prompt_template.batch(input_payloads)

    completions, completion_metadata = do_inference(
        chat_model=chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
        usage_tracker=usage_tracker,
        usage_phase=usage_phase,
        usage_model_spec=model,
        return_metadata=True,
    )
    hit_token_limit = _record_generation_output_limit_events(
        metadata=completion_metadata,
        case_ids=case_ids,
        field="completion",
        model_spec=model,
        limit_event_tracker=limit_event_tracker,
    )
    df_outputs = pd.DataFrame(
        data={
            "completion": completions,
            "instruction_index": case_ids,
            "generation_prompt_truncated": prompt_truncated,
            "generation_output_finish_reason": [
                metadata_row.get("finish_reason")
                for metadata_row in completion_metadata
            ],
            "generation_output_hit_token_limit": hit_token_limit,
        },
    )
    return df_outputs


def _set_temperature_on_model(chat_model, temperature: float) -> None:
    if hasattr(chat_model, "set_temperature"):
        chat_model.set_temperature(temperature)
        return
    if hasattr(chat_model, "temperature"):
        chat_model.temperature = temperature


def _infer_grouped_by_temperature(
    *,
    model_spec: str,
    provider: str,
    max_tokens: int | None,
    model_kwargs: dict,
    base_model,
    inputs: list,
    temperatures: list[float],
    use_tqdm: bool,
    usage_tracker=None,
    usage_phase: str | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    outputs: list[str] = [""] * len(inputs)
    outputs_metadata: list[dict[str, Any]] = [{} for _ in inputs]
    groups: dict[float, list[int]] = {}
    for idx, temp in enumerate(temperatures):
        groups.setdefault(float(temp), []).append(idx)

    for temp in sorted(groups.keys()):
        idxs = groups[temp]
        group_inputs = [inputs[i] for i in idxs]

        if provider in {"VLLM", "LlamaCpp"}:
            _set_temperature_on_model(base_model, temp)
            group_model = base_model
        else:
            group_model = make_model(
                model_spec, max_tokens=max_tokens, temperature=temp, **model_kwargs
            )

        group_outs, group_metadata = do_inference(
            chat_model=group_model,
            inputs=group_inputs,
            use_tqdm=use_tqdm,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
            usage_model_spec=model_spec,
            return_metadata=True,
        )
        for i, out, metadata_row in zip(idxs, group_outs, group_metadata, strict=True):
            outputs[i] = out
            outputs_metadata[i] = metadata_row

    return outputs, outputs_metadata


def generate_multiturn(
    questions: pd.DataFrame,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 8192,
    use_tqdm: bool = True,
    temperature_config: dict[str, float] | None = None,
    usage_tracker=None,
    usage_phase: str | None = None,
    limit_event_tracker: LimitEventTracker | None = None,
    strip_thinking_in_turn_1_carryover: bool = True,
    **model_kwargs,
) -> pd.DataFrame:
    """Generate two-turn completions for MT-Bench style questions."""
    provider = model.split("/")[0]
    use_category_temperatures = temperature_config is not None
    local_provider = provider in {"VLLM", "LlamaCpp"}

    if use_category_temperatures and local_provider:
        chat_model = make_model(
            model,
            max_tokens=max_tokens,
            temperature=0.0,
            limit_event_tracker=limit_event_tracker,
            limit_event_stage="generation_model_init",
            limit_event_model_spec=model,
            **model_kwargs,
        )
    else:
        chat_model = make_model(
            model,
            max_tokens=max_tokens,
            limit_event_tracker=limit_event_tracker,
            limit_event_stage="generation_model_init",
            limit_event_model_spec=model,
            **model_kwargs,
        )

    system_prompt = "You are a helpful assistant."
    idxs = questions.index.tolist()
    temperatures: list[float] = []
    if use_category_temperatures:
        temperatures = [
            temperature_config.get(str(questions.loc[idx].get("category") or ""), 0.7)
            for idx in idxs
        ]

    turn1_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{user_prompt}")]
    )
    turn1_prompt_truncated: list[bool] = []
    turn1_payloads = []
    for question_id, row in questions.iterrows():
        truncated_turn_1, was_truncated = truncate_with_metadata(
            row["turn_1"],
            max_len=truncate_input_chars,
            tracker=limit_event_tracker,
            kind="generation_input_char_truncation",
            stage="generation_input",
            field="turn_1",
            case_id=question_id,
            model_spec=model,
        )
        turn1_prompt_truncated.append(was_truncated)
        turn1_payloads.append({"user_prompt": truncated_turn_1})
    turn1_inputs = turn1_template.batch(turn1_payloads)

    if use_category_temperatures:
        completions_turn_1, turn1_metadata = _infer_grouped_by_temperature(
            model_spec=model,
            provider=provider,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            base_model=chat_model,
            inputs=turn1_inputs,
            temperatures=temperatures,
            use_tqdm=use_tqdm,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
        )
    else:
        completions_turn_1, turn1_metadata = do_inference(
            chat_model=chat_model,
            inputs=turn1_inputs,
            use_tqdm=use_tqdm,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
            usage_model_spec=model,
            return_metadata=True,
        )
    turn1_hit_token_limit = _record_generation_output_limit_events(
        metadata=turn1_metadata,
        case_ids=idxs,
        field="completion_turn_1",
        model_spec=model,
        limit_event_tracker=limit_event_tracker,
    )

    turn2_inputs = []
    turn2_turn1_truncated: list[bool] = []
    turn2_answer_truncated: list[bool] = []
    turn2_prompt_truncated: list[bool] = []
    turn2_turn1_answer_thinking_stripped: list[bool] = []
    for (question_id, row), t1_answer in zip(
        questions.iterrows(), completions_turn_1, strict=True
    ):
        if row["turn_2"] is None:
            turn2_turn1_truncated.append(False)
            turn2_answer_truncated.append(False)
            turn2_prompt_truncated.append(False)
            turn2_turn1_answer_thinking_stripped.append(False)
            turn2_inputs.append(
                turn1_template.invoke({"user_prompt": "No follow-up question."})
            )
        else:
            multi_turn_template = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("user", "{turn_1}"),
                    ("assistant", "{turn_1_answer}"),
                    ("user", "{turn_2}"),
                ]
            )
            truncated_turn_1, turn1_was_truncated = truncate_with_metadata(
                row["turn_1"],
                max_len=truncate_input_chars,
                tracker=limit_event_tracker,
                kind="generation_input_char_truncation",
                stage="generation_input",
                field="turn_1_for_turn_2",
                case_id=question_id,
                model_spec=model,
            )
            # Strip <think>...</think> from the turn-1 answer before the
            # character cap fires. Mirrors what the Qwen3 chat template does
            # natively for historical assistant turns; applying it here
            # ensures a 30K-char cap lands on the visible answer rather than
            # deep inside a runaway reasoning block, which would silently
            # destroy the </think> closer and force the whole thinking
            # fragment into the turn-2 prompt.
            t1_answer_str = str(t1_answer)
            if strip_thinking_in_turn_1_carryover:
                t1_answer_str, thinking_stripped = strip_thinking_tags_with_metadata(
                    t1_answer_str
                )
            else:
                thinking_stripped = False
            turn2_turn1_answer_thinking_stripped.append(thinking_stripped)
            truncated_turn_1_answer, answer_was_truncated = truncate_with_metadata(
                t1_answer_str,
                max_len=truncate_input_chars,
                tracker=limit_event_tracker,
                kind="generation_input_char_truncation",
                stage="generation_input",
                field="turn_1_answer",
                case_id=question_id,
                model_spec=model,
            )
            truncated_turn_2, turn2_was_truncated = truncate_with_metadata(
                row["turn_2"],
                max_len=truncate_input_chars,
                tracker=limit_event_tracker,
                kind="generation_input_char_truncation",
                stage="generation_input",
                field="turn_2",
                case_id=question_id,
                model_spec=model,
            )
            turn2_turn1_truncated.append(turn1_was_truncated)
            turn2_answer_truncated.append(answer_was_truncated)
            turn2_prompt_truncated.append(turn2_was_truncated)
            turn2_inputs.append(
                multi_turn_template.invoke(
                    {
                        "turn_1": truncated_turn_1,
                        "turn_1_answer": truncated_turn_1_answer,
                        "turn_2": truncated_turn_2,
                    }
                )
            )

    if use_category_temperatures:
        completions_turn_2, turn2_metadata = _infer_grouped_by_temperature(
            model_spec=model,
            provider=provider,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            base_model=chat_model,
            inputs=turn2_inputs,
            temperatures=temperatures,
            use_tqdm=use_tqdm,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
        )
    else:
        completions_turn_2, turn2_metadata = do_inference(
            chat_model=chat_model,
            inputs=turn2_inputs,
            use_tqdm=use_tqdm,
            usage_tracker=usage_tracker,
            usage_phase=usage_phase,
            usage_model_spec=model,
            return_metadata=True,
        )
    turn2_hit_token_limit = _record_generation_output_limit_events(
        metadata=turn2_metadata,
        case_ids=idxs,
        field="completion_turn_2",
        model_spec=model,
        limit_event_tracker=limit_event_tracker,
    )

    return pd.DataFrame(
        data={
            "instruction_index": idxs,
            "completion_turn_1": completions_turn_1,
            "completion_turn_2": completions_turn_2,
            "generation_turn_1_prompt_truncated": turn1_prompt_truncated,
            "generation_turn_1_finish_reason": [
                metadata_row.get("finish_reason") for metadata_row in turn1_metadata
            ],
            "generation_turn_1_hit_token_limit": turn1_hit_token_limit,
            "generation_turn_2_turn_1_prompt_truncated": turn2_turn1_truncated,
            "generation_turn_2_turn_1_answer_truncated": turn2_answer_truncated,
            "generation_turn_2_turn_1_answer_thinking_stripped": (
                turn2_turn1_answer_thinking_stripped
            ),
            "generation_turn_2_prompt_truncated": turn2_prompt_truncated,
            "generation_turn_2_finish_reason": [
                metadata_row.get("finish_reason") for metadata_row in turn2_metadata
            ],
            "generation_turn_2_hit_token_limit": turn2_hit_token_limit,
        },
    )


def generate_base(
    instructions: pd.Series,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 32768,
    use_tqdm: bool = False,
    usage_tracker=None,
    usage_phase: str | None = None,
    limit_event_tracker: LimitEventTracker | None = None,
    **engine_kwargs,
) -> pd.DataFrame:
    model_spec = model
    model = make_model(
        model_spec,
        max_tokens=max_tokens,
        limit_event_tracker=limit_event_tracker,
        limit_event_stage="generation_model_init",
        limit_event_model_spec=model_spec,
        **engine_kwargs,
    )

    prompt_truncated: list[bool] = []
    case_ids = instructions.index.tolist()
    inputs = []
    for case_id, instruction in zip(case_ids, instructions, strict=True):
        truncated_instruction, was_truncated = truncate_with_metadata(
            instruction,
            max_len=truncate_input_chars,
            tracker=limit_event_tracker,
            kind="generation_input_char_truncation",
            stage="generation_input",
            field="instruction",
            case_id=case_id,
            model_spec=model_spec,
        )
        prompt_truncated.append(was_truncated)
        inputs.append(truncated_instruction)

    completions, completion_metadata = do_inference(
        chat_model=model,
        inputs=inputs,
        use_tqdm=use_tqdm,
        usage_tracker=usage_tracker,
        usage_phase=usage_phase,
        usage_model_spec=model_spec,
        return_metadata=True,
    )
    hit_token_limit = _record_generation_output_limit_events(
        metadata=completion_metadata,
        case_ids=case_ids,
        field="completion",
        model_spec=model_spec,
        limit_event_tracker=limit_event_tracker,
    )

    df_outputs = pd.DataFrame(
        data={
            "completion": completions,
            "instruction_index": case_ids,
            "generation_prompt_truncated": prompt_truncated,
            "generation_output_finish_reason": [
                metadata_row.get("finish_reason")
                for metadata_row in completion_metadata
            ],
            "generation_output_hit_token_limit": hit_token_limit,
        },
    )

    return df_outputs
