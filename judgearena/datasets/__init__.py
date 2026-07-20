from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from judgearena.log import get_logger

if TYPE_CHECKING:
    from judgearena.tasks.schema import ResolvedTaskSpec

logger = get_logger(__name__)


def load_instructions(
    dataset: str | ResolvedTaskSpec, n_instructions: int | None = None
) -> pd.DataFrame:
    """Load instructions by task ID or an already-resolved task definition."""
    dataset_id = dataset if isinstance(dataset, str) else dataset.task
    if isinstance(dataset, str):
        from judgearena.tasks.registry import get_packaged_task

        resolved_task = get_packaged_task(dataset)
    else:
        resolved_task = dataset
    if resolved_task is not None:
        from judgearena import utils as judgearena_utils
        from judgearena.datasets.registry import resolve_dataset_adapter

        adapter = resolve_dataset_adapter(resolved_task.spec.dataset.adapter)
        df_instructions = adapter.load_instructions(
            resolved_task, judgearena_utils.data_root / "tables"
        )

    else:
        raise ValueError(f"Unsupported instruction dataset {dataset_id!r}.")

    df_instructions = df_instructions.set_index("instruction_index").sort_index()
    logger.info("Loaded %d instructions for %s.", len(df_instructions), dataset_id)
    if n_instructions is None:
        n_instructions = len(df_instructions)
    return df_instructions.head(n_instructions)


if __name__ == "__main__":
    instructions = load_instructions(dataset="alpaca-eval")
    print(instructions)
