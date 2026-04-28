import pandas as pd

from judgearena.instruction_dataset.arena_hard import (
    download_arena_hard,
    is_arena_hard_dataset,
)
from judgearena.instruction_dataset.m_arenahard import (
    load_m_arenahard,
    split_m_arena_hard_dataset,
)
from judgearena.log import get_logger
from judgearena.utils import data_root, download_hf, read_df

logger = get_logger(__name__)


def load_instructions(dataset: str, n_instructions: int | None = None) -> pd.DataFrame:
    if dataset == "mt-bench":
        from judgearena.instruction_dataset.mt_bench import load_mt_bench

        df_instructions = load_mt_bench()

    elif (parsed := split_m_arena_hard_dataset(dataset)) is not None:
        version_key, lang_or_subset = parsed
        logger.info(
            "Loading %s with language specification set to %s",
            version_key,
            lang_or_subset,
        )
        df_instructions = load_m_arenahard(
            local_path=data_root, version=version_key, language=lang_or_subset
        )
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
        assert dataset in [
            "alpaca-eval",
            "arena-hard-v0.1",
            "arena-hard-v2.0",
        ]
        local_path_tables = data_root / "tables"
        if is_arena_hard_dataset(dataset):
            download_arena_hard(dataset=dataset, local_tables_path=local_path_tables)
        else:
            download_hf(name=dataset, local_path=local_path_tables)
        df_instructions = read_df(local_path_tables / "instructions" / f"{dataset}.csv")

    df_instructions = df_instructions.set_index("instruction_index").sort_index()
    logger.info("Loaded %d instructions for %s.", len(df_instructions), dataset)
    if n_instructions is None:
        n_instructions = len(df_instructions)
    return df_instructions.head(n_instructions)


if __name__ == "__main__":
    instructions = load_instructions(dataset="alpaca-eval")
    print(instructions)
