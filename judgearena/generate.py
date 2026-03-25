import pandas as pd
from langchain.prompts import ChatPromptTemplate

from judgearena.utils import (
    do_inference,
    make_model,
)


def truncate(s: str, max_len: int | None = None):
    if max_len is not None:
        return s[:max_len]
    else:
        return s


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
