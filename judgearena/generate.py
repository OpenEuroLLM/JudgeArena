import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from judgearena.utils import (
    do_inference,
    make_model,
    truncate,
)


def generate_instructions(
    instructions: pd.Series,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 32768,
    use_tqdm: bool = True,
    system_prompt: str | None = None,
    **engine_kwargs,
) -> pd.DataFrame:
    chat_model = make_model(model, max_tokens=max_tokens, **engine_kwargs)

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
) -> list[str]:
    outputs: list[str] = [""] * len(inputs)
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

        group_outs = do_inference(
            chat_model=group_model,
            inputs=group_inputs,
            use_tqdm=use_tqdm,
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
    **model_kwargs,
) -> pd.DataFrame:
    """Generate two-turn completions for MT-Bench style questions."""
    provider = model.split("/")[0]
    use_category_temperatures = temperature_config is not None
    local_provider = provider in {"VLLM", "LlamaCpp"}

    if use_category_temperatures and local_provider:
        chat_model = make_model(
            model, max_tokens=max_tokens, temperature=0.0, **model_kwargs
        )
    else:
        chat_model = make_model(model, max_tokens=max_tokens, **model_kwargs)

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
    turn1_inputs = turn1_template.batch(
        [
            {"user_prompt": truncate(row["turn_1"], max_len=truncate_input_chars)}
            for _, row in questions.iterrows()
        ]
    )

    if use_category_temperatures:
        completions_turn_1 = _infer_grouped_by_temperature(
            model_spec=model,
            provider=provider,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            base_model=chat_model,
            inputs=turn1_inputs,
            temperatures=temperatures,
            use_tqdm=use_tqdm,
        )
    else:
        completions_turn_1 = do_inference(
            chat_model=chat_model,
            inputs=turn1_inputs,
            use_tqdm=use_tqdm,
        )

    turn2_inputs = []
    for (_, row), t1_answer in zip(
        questions.iterrows(), completions_turn_1, strict=True
    ):
        if row["turn_2"] is None:
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
            turn2_inputs.append(
                multi_turn_template.invoke(
                    {
                        "turn_1": truncate(row["turn_1"], max_len=truncate_input_chars),
                        "turn_1_answer": truncate(
                            str(t1_answer), max_len=truncate_input_chars
                        ),
                        "turn_2": truncate(row["turn_2"], max_len=truncate_input_chars),
                    }
                )
            )

    if use_category_temperatures:
        completions_turn_2 = _infer_grouped_by_temperature(
            model_spec=model,
            provider=provider,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            base_model=chat_model,
            inputs=turn2_inputs,
            temperatures=temperatures,
            use_tqdm=use_tqdm,
        )
    else:
        completions_turn_2 = do_inference(
            chat_model=chat_model,
            inputs=turn2_inputs,
            use_tqdm=use_tqdm,
        )

    return pd.DataFrame(
        data={
            "instruction_index": idxs,
            "completion_turn_1": completions_turn_1,
            "completion_turn_2": completions_turn_2,
        },
    )


def generate_base(
    instructions: pd.Series,
    model: str,
    truncate_input_chars: int | None = 8192,
    max_tokens: int | None = 32768,
    use_tqdm: bool = False,
    **engine_kwargs,
) -> pd.DataFrame:
    model = make_model(model, max_tokens=max_tokens, **engine_kwargs)

    inputs = [
        truncate(instruction, max_len=truncate_input_chars)
        for instruction in instructions
    ]

    completions = model.batch(
        inputs=inputs,
        max_tokens=max_tokens,
    )
    completions = [x.content if hasattr(x, "content") else x for x in completions]

    df_outputs = pd.DataFrame(
        data={
            "completion": completions,
            "instruction_index": instructions.index.tolist(),
        },
    )

    return df_outputs
