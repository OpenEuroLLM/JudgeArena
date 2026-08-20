import pandas as pd

from judgearena.log import get_logger
from judgearena.tasks.registry import get_packaged_task

logger = get_logger(__name__)


def load_instructions(dataset: str, n_instructions: int | None = None) -> pd.DataFrame:
    resolved_task = get_packaged_task(dataset)
    if resolved_task is not None:
        from judgearena import utils as judgearena_utils
        from judgearena.datasets.registry import resolve_dataset_adapter

        adapter = resolve_dataset_adapter(resolved_task.spec.dataset.adapter)
        df_instructions = adapter.load_instructions(
            resolved_task, judgearena_utils.data_root / "tables"
        )

    else:
        raise ValueError(f"Unsupported instruction dataset {dataset!r}.")

    df_instructions = df_instructions.set_index("instruction_index").sort_index()
    logger.info("Loaded %d instructions for %s.", len(df_instructions), dataset)
    if n_instructions is None:
        n_instructions = len(df_instructions)
    return df_instructions.head(n_instructions)


if __name__ == "__main__":
    instructions = load_instructions(dataset="alpaca-eval")
    print(instructions)
