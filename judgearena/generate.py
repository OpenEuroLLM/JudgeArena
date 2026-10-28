import pandas as pd
from langchain_core.messages import convert_to_messages
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate

from judgearena.cache.inference import CompletionInferenceCache
from judgearena.datasets.data import (
    _require_columns,
    validate_messages,
    validate_user_turns,
)
from judgearena.models import PreparedModel, do_inference, prepare_model
from judgearena.utils import strip_thinking_tags, truncate


def _with_system_prompt(messages, system_prompt):
    if system_prompt is None:
        return list(messages)
    history = messages[1:] if messages and messages[0]["role"] == "system" else messages
    return [{"role": "system", "content": system_prompt}, *history]


def _chat_input(
    messages,
    *,
    system_prompt=None,
    truncate_input_chars=None,
    strip_thinking=False,
):
    validate_messages(messages, ending="user")
    rendered = []
    for message in _with_system_prompt(messages, system_prompt):
        content = message["content"]
        if strip_thinking and message["role"] == "assistant":
            content = strip_thinking_tags(content)
        rendered.append(
            (message["role"], truncate(content, max_len=truncate_input_chars))
        )
    return ChatPromptValue(messages=convert_to_messages(rendered))


def generate_messages(
    instructions: pd.Series,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 8192,
    use_tqdm: bool = True,
    system_prompt: str | None = None,
    inference_cache: CompletionInferenceCache | None = None,
    **model_kwargs,
) -> pd.DataFrame:
    """Generate one answer per supplied conversation, without extending other rows.

    Each Series value is a list of role/content messages ending with the user's
    request. Its index supplies the completion and cache IDs. No system message
    is added unless supplied; an explicit system_prompt replaces an existing
    leading system message. Message text is literal, including braces.
    """
    chat_model = prepare_model(
        model, max_tokens=max_tokens, cache=inference_cache, **model_kwargs
    )
    inputs = [
        _chat_input(
            messages,
            system_prompt=system_prompt,
            truncate_input_chars=truncate_input_chars,
        )
        for messages in instructions
    ]
    completions = do_inference(
        chat_model=chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
        stage="generation",
        cache_row_metadata=[
            {"instruction_id": str(index)} for index in instructions.index
        ],
    )
    return pd.DataFrame(
        {
            "instruction_index": instructions.index,
            "completion": completions,
            "messages": [
                _with_system_prompt(history, system_prompt)
                + [{"role": "assistant", "content": completion}]
                for history, completion in zip(instructions, completions, strict=True)
            ],
        }
    )


def generate_instructions(
    instructions: pd.Series,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 32768,
    use_tqdm: bool = True,
    system_prompt: str | None = None,
    inference_cache: CompletionInferenceCache | None = None,
    **engine_kwargs,
) -> pd.DataFrame:
    chat_model = prepare_model(
        model,
        max_tokens=max_tokens,
        cache=inference_cache,
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

    inputs = prompt_template.batch(
        [
            {
                "user_prompt": truncate(user_prompt, max_len=truncate_input_chars),
            }
            for user_prompt in instructions
        ]
    )

    completions = do_inference(
        chat_model=chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
        stage="generation",
        cache_row_metadata=[
            {"instruction_id": str(index)} for index in instructions.index
        ],
    )
    df_outputs = pd.DataFrame(
        data={
            "completion": completions,
            "instruction_index": instructions.index.tolist(),
        },
    )
    return df_outputs


def _set_temperature_on_model(chat_model, temperature: float) -> None:
    if hasattr(chat_model, "set_temperature"):
        chat_model.set_temperature(temperature)
        return
    if hasattr(chat_model, "temperature"):
        chat_model.temperature = temperature


def _materialize_at_temperature(base_model: PreparedModel, temperature: float):
    model = base_model.materialize()
    _set_temperature_on_model(model, temperature)
    return model


def _infer_grouped_by_temperature(
    *,
    model_spec: str,
    max_tokens: int | None,
    model_kwargs: dict,
    base_model: PreparedModel | None,
    inputs: list,
    instruction_ids: list[str],
    temperatures: list[float],
    use_tqdm: bool,
    inference_cache: CompletionInferenceCache | None,
) -> list[str]:
    outputs: list[str] = [""] * len(inputs)
    groups: dict[float, list[int]] = {}
    for idx, temp in enumerate(temperatures):
        groups.setdefault(float(temp), []).append(idx)

    for temp in sorted(groups.keys()):
        idxs = groups[temp]
        group_inputs = [inputs[i] for i in idxs]

        group_model = prepare_model(
            model_spec,
            max_tokens=max_tokens,
            temperature=temp,
            cache=inference_cache,
            **model_kwargs,
        )
        if base_model is not None:
            group_model.factory = lambda temperature=temp: _materialize_at_temperature(
                base_model, temperature
            )

        group_outs = do_inference(
            chat_model=group_model,
            inputs=group_inputs,
            use_tqdm=use_tqdm,
            stage="generation",
            cache_row_metadata=[
                {"instruction_id": instruction_ids[index]} for index in idxs
            ],
        )
        for i, out in zip(idxs, group_outs, strict=True):
            outputs[i] = out

    return outputs


def generate_multiturn(
    questions: pd.DataFrame,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 8192,
    use_tqdm: bool = True,
    temperature_config: dict[str, float] | None = None,
    fallback_temperature: float = 0.0,
    system_prompt: str | None = None,
    strip_thinking_before_history: bool = False,
    inference_cache: CompletionInferenceCache | None = None,
    **model_kwargs,
) -> pd.DataFrame:
    """Generate each row's user_turns sequentially, retaining full model histories."""
    _require_columns(questions, {"user_turns"}, "Conversation generation")
    turns = questions["user_turns"].tolist()
    for row in turns:
        validate_user_turns(row)
    provider = model.split("/")[0]
    use_category_temperatures = temperature_config is not None
    local_provider = provider in {"VLLM", "LlamaCpp"}

    base_model = (
        prepare_model(
            model,
            max_tokens=max_tokens,
            temperature=0.0,
            **model_kwargs,
        )
        if use_category_temperatures and local_provider
        else None
    )
    chat_model = prepare_model(
        model,
        max_tokens=max_tokens,
        cache=inference_cache,
        **model_kwargs,
    )

    idxs = questions.index.tolist()
    temperatures: list[float] = []
    if use_category_temperatures:
        temperatures = [
            temperature_config.get(
                str(questions.loc[idx].get("category") or ""), fallback_temperature
            )
            for idx in idxs
        ]

    histories: list[list[dict[str, str]]] = [[] for _ in turns]
    for turn in range(max(map(len, turns), default=0)):
        active = [i for i, row in enumerate(turns) if turn < len(row)]
        inputs = []
        for i in active:
            histories[i].append({"role": "user", "content": turns[i][turn]})
            inputs.append(
                _chat_input(
                    histories[i],
                    system_prompt=system_prompt,
                    truncate_input_chars=truncate_input_chars,
                    strip_thinking=strip_thinking_before_history,
                )
            )
        ids = [f"{idxs[i]}:turn-{turn + 1}" for i in active]
        if use_category_temperatures:
            answers = _infer_grouped_by_temperature(
                model_spec=model,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
                base_model=base_model,
                inputs=inputs,
                instruction_ids=ids,
                temperatures=[temperatures[i] for i in active],
                use_tqdm=use_tqdm,
                inference_cache=inference_cache,
            )
        else:
            answers = do_inference(
                chat_model=chat_model,
                inputs=inputs,
                use_tqdm=use_tqdm,
                stage="generation",
                cache_row_metadata=[{"instruction_id": i} for i in ids],
            )
        for i, answer in zip(active, answers, strict=True):
            histories[i].append({"role": "assistant", "content": str(answer)})
    return pd.DataFrame(
        {
            "instruction_index": idxs,
            "messages": histories,
            "completion": [row[-1]["content"] for row in histories],
        }
    )


def generate_base(
    instructions: pd.Series,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 32768,
    use_tqdm: bool = False,
    inference_cache: CompletionInferenceCache | None = None,
    **engine_kwargs,
) -> pd.DataFrame:
    chat_model = prepare_model(
        model,
        max_tokens=max_tokens,
        cache=inference_cache,
        **engine_kwargs,
    )

    inputs = [
        truncate(instruction, max_len=truncate_input_chars)
        for instruction in instructions
    ]

    completions = do_inference(
        chat_model=chat_model,
        inputs=inputs,
        stage="generation",
        cache_row_metadata=[
            {"instruction_id": str(index)} for index in instructions.index
        ],
    )

    df_outputs = pd.DataFrame(
        data={
            "completion": completions,
            "instruction_index": instructions.index.tolist(),
        },
    )

    return df_outputs
