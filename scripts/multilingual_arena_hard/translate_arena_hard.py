"""
Generate dataset for ArenaHard-EU by translating all instructions from ArenaHard to 34 languages.

The 34 languages include all official EU languages as well as co-official languages in Member States,
languages of candidate EU Members and Scandinavian countries.

The 34 languages are the ones targeted by OpenEuroLLM for the first release:
https://github.com/OpenEuroLLM/training-data-catalogue/blob/main/languages

Dataset is uploaded at:
https://huggingface.co/datasets/openeurollm/ArenaHard-EU/
"""

from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain.prompts import ChatPromptTemplate

from judgearena.instruction_dataset import load_instructions
from judgearena.utils import do_inference, make_model
from judgearena.utils import set_langchain_cache

# set_langchain_cache()

dataset_name = "openeurollm/ArenaHard-EU-v0-bis"

"""
TODOs:
- test with JudgeArena
Done:
- load AH
- call translate
- check translate quality in French
- dump data to disk
- update to HF
"""
languages = [
    ("fra", "French"),
    # ("spa", "Spanish"),
    # ("deu", "German"),
    # ("bul", "Bulgarian"),
    # ("ces", "Czech"),
    # ("dan", "Danish"),
    # ("ell", "Greek"),
    # # ("eng", "English"),
    # ("est", "Estonian"),
    ("fin", "Finnish"),
    # ("gle", "Irish"),
    # ("hrv", "Croatian"),
    # ("hun", "Hungarian"),
    # ("ita", "Italian"),
    # ("lav", "Latvian"),
    # ("lit", "Lithuanian"),
    # ("mlt", "Maltese"),
    # ("nld", "Dutch"),
    # ("pol", "Polish"),
    # ("por", "Portuguese"),
    # ("ron", "Romanian"),
    # ("slk", "Slovak"),
    # ("slv", "Slovene"),
    # ("swe", "Swedish"),
    # ("cat", "Catalan"),
    # ("eus", "Basque"),
    # ("glg", "Galician"),
    # ("bos", "Bosnian"),
    # ("kat", "Georgian"),
    # ("mkd", "Macedonian"),
    # ("sqi", "Albanian"),
    # ("srp", "Serbian"),
    # ("tur", "Turkish"),
    # ("ukr", "Ukrainian"),
    # ("isl", "Icelandic"),
    # ("nor", "Norwegian"),
]

translator_model = "OpenRouter/openai/gpt-5"
# translator_model = "OpenRouter/deepseek/deepseek-chat-v3.1"
n_instructions = 10
df_instructions = load_instructions(
    "arena-hard",
    n_instructions=n_instructions,
)
# languages = [("fra", "French")]


def save_df(
    language_code: str,
    df_instructions: pd.DataFrame,
    translated_instructions: list[str],
):
    translation_path = (
        f"data/ArenaHard-EU/{language_code}/data-{len(translated_instructions)}.parquet"
    )
    print(f"Saving translations to {translation_path}")
    Path(translation_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "question_id": df_instructions.loc[:, "question_id"],
            "category": df_instructions.loc[:, "category"],
            "domain": df_instructions.loc[:, "domain"],
            "prompt": translated_instructions,
            "lang": language_code,
        }
    )
    df.to_parquet(translation_path, index=False)


def generate_translations():
    for language_code, target_language in languages:
        system_prompt = (
            f"You are an expert translator from English to {target_language}."
        )

        user_prompt_template = f"""\
        Your task is to translate the following text from English to {target_language}.
        
        Important:
        * Do not answer the instruction, just translate it.
        * If any code is present in the instruction, **do not change it and do not translate it**.
        * Keep a formality adapted for the target language, most language would not use "Vous" or "Sie" when talking to a chatbot.
        
        # Input text
        {{instruction}}
        
        # Output text (just output the translation and nothing else as your ouput will be parsed)\n         
        """

        judge_chat_model = make_model(translator_model, max_tokens=32768)
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", user_prompt_template)]
        )

        inputs = prompt_template.batch(
            [
                {"instruction": instruction}
                for instruction in df_instructions.loc[:, "instruction"]
            ]
        )
        print(f"Start translation to {target_language} ({len(inputs)} documents).")
        translated_instructions = do_inference(
            chat_model=judge_chat_model,
            inputs=inputs,
            use_tqdm=False,
        )

        translation_path = f"data/ArenaHard-EU/{language_code}/data.parquet"
        Path(translation_path).parent.mkdir(parents=True, exist_ok=True)

        print("\n--------------------\n".join(translated_instructions))
        save_df(
            language_code=language_code,
            df_instructions=df_instructions,
            translated_instructions=translated_instructions,
        )

    # saving english as well
    save_df(
        language_code="eng",
        df_instructions=df_instructions,
        translated_instructions=df_instructions.loc[:, "instruction"],
    )


# upload to HuggingFace
def upload_hugging_face():

    # Upload each language as a separate config
    for lang, _ in languages + [("eng", "English")]:
        print(f"Uploading {lang}...")

        # Load the parquet file for this language
        df = pd.read_parquet(f"data/ArenaHard-EU/{lang}/data-10.parquet")
        dataset = Dataset.from_pandas(df)

        # Push to hub with config_name
        dataset.push_to_hub(
            repo_id=dataset_name,
            config_name=lang,  # Each language becomes a separate config
            split="train",  # Specify the split (train/validation/test)
            private=False,
        )

    print("All languages uploaded successfully!")


generate_translations()
upload_hugging_face()
