from pathlib import Path

import pandas as pd
from datasets import Dataset

from judgearena.utils import do_inference, make_model, set_langchain_cache

set_langchain_cache()

dataset_name = "geoalgo/multilingual-fluency"
model = "OpenRouter/openai/gpt-5-mini"
n_langs = 44
n_context_to_generate = 200
languages = [
    # sorted by number of speakers by claude
    "English",
    "Mandarin Chinese",
    "Hindi",
    "Spanish",
    "French",
    "Standard Arabic",
    "Bengali",
    "Russian",
    "Portuguese",
    "Indonesian",
    "Japanese",
    "German",
    "Turkish",
    "Italian",
    "Ukrainian",
    "Polish",
    "Romanian",
    "Dutch",
    "Greek",
    "Hungarian",
    "Czech",
    "Catalan",
    "Bulgarian",
    "Swedish",
    "Danish",
    "Finnish",
    "Slovak",
    "Norwegian",
    "Croatian",
    "Georgian",
    "Lithuanian",
    "Slovene",
    "Latvian",
    "Albanian",
    "Macedonian",
    "Estonian",
    "Basque",
    "Galician",
    "Bosnian",
    "Serbian",
    "Icelandic",
    "Maltese",
    "Irish",
]


def make_fluency_prompt(lang: str, n_sentences_to_generate: int) -> str:
    prompt = f"""\
Please generate a list of {n_sentences_to_generate} sentences that could be used to compare different LLM pretrain models.

For each sentence, I will compare the completion of a base pre-trained models (**not an instruction tuned model**) with an LLM-judge to evaluate the fluency in a generative settings.

* All sentences should be in {lang}.
* All sentences should cut out after a random number of tokens (between 5 and 10 tokens).
* Include sentences with general sentences, history, economics, math and programming.

Put the {n_sentences_to_generate} sentences in a csv with columns "type", "sentence" where type is one of "general sentences", history, economics, math and programming".

Put quotes for the sentences so that the csv can be read in pandas.

Make sure there is exactly {n_sentences_to_generate} items.
"""
    return prompt


def generate_contexts(
    model: str,
    languages: list[str],
    n_sentences_to_generate: int,
    ignore_cache: bool = False,
):

    for target_language in languages:
        data_path = Path(__file__).parent / "data" / f"{target_language}-contexts.csv"
        data_path.parent.mkdir(parents=True, exist_ok=True)

        if not data_path.exists() or ignore_cache:
            judge_chat_model = make_model(model, max_tokens=65536)

            print(
                f"Generating {n_sentences_to_generate} contexts for {target_language}."
            )
            output = do_inference(
                chat_model=judge_chat_model,
                inputs=[
                    make_fluency_prompt(
                        lang=target_language,
                        n_sentences_to_generate=n_sentences_to_generate,
                    )
                ],
                use_tqdm=False,
            )

            if len(output[0]) < 10:
                print(
                    f"Short completion for language {target_language} ({len(output[0])} chars) not storing it."
                )
            else:
                with open(data_path, "w") as f:
                    f.write(output[0])

                print(f"Try to load {data_path}")
                print("Context generated:")
                print(pd.read_csv(data_path).head().to_string())


def upload_hugging_face(languages: list[str]):
    # Upload each language as a separate config to have subsets
    for language in languages:
        print(f"Uploading {language}...")

        # Load the parquet file for this language
        df = pd.read_csv(Path(__file__).parent / f"data/{language}-contexts.csv")
        dataset = Dataset.from_pandas(df)

        # Push to hub with config_name
        dataset.push_to_hub(
            repo_id=dataset_name,
            config_name=language,  # Each language becomes a separate config
            split="train",  # Specify the split (train/validation/test)
            private=False,
        )

    print("All languages uploaded successfully!")


generate_contexts(
    model=model,
    languages=languages[:n_langs],
    n_sentences_to_generate=n_context_to_generate,
)
upload_hugging_face(languages=languages[:n_langs])
