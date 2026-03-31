import pandas as pd

from judgearena.instruction_dataset.m_arenahard import load_m_arenahard
from judgearena.utils import data_root, download_hf, read_df


def load_instructions(dataset: str, n_instructions: int | None = None) -> pd.DataFrame:
    if dataset == "mt-bench":
        from judgearena.instruction_dataset.mt_bench import load_mt_bench

        df_instructions = load_mt_bench()

    elif "m-arena-hard" in dataset:
        if dataset == "m-arena-hard":
            language = None
        else:
            # read the suffix part "m-arena-hard-EU" -> "EU"
            language = dataset.split("-")[-1]
            assert language in [
                None,
                "ar",
                "cs",
                "de",
                "el",
                "en",
                "es",
                "fa",
                "fr",
                "he",
                "hi",
                "id",
                "it",
                "ja",
                "ko",
                "nl",
                "pl",
                "pt",
                "ro",
                "ru",
                "tr",
                "uk",
                "vi",
                "zh",
                "EU",
            ]
        print(f"Loading m-arena-hard with language specification set to {language}")
        df_instructions = load_m_arenahard(local_path=data_root, language=language)

        # sort by question_id, then language so that we get multiple languages if we truncate
        df_instructions.sort_values(["question_id", "lang"], inplace=True)
        df_instructions.rename(
            {
                "question_id": "instruction_index",
                "prompt": "instruction",
            },
            axis=1,
            inplace=True,
        )

    else:
        assert dataset in ["alpaca-eval", "arena-hard"]
        local_path_tables = data_root / "tables"
        download_hf(name=dataset, local_path=local_path_tables)
        df_instructions = read_df(local_path_tables / "instructions" / f"{dataset}.csv")

    df_instructions = df_instructions.set_index("instruction_index").sort_index()
    print(f"Loaded {len(df_instructions)} instructions for {dataset}.")
    if n_instructions is None:
        n_instructions = len(df_instructions)
    return df_instructions.head(n_instructions)


if __name__ == "__main__":
    instructions = load_instructions(dataset="alpaca-eval")
    print(instructions)
