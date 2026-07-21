import pandas as pd

from judgearena.log import get_logger
from judgearena.tasks.schema import ResolvedTaskSpec

logger = get_logger(__name__)


def _resolve_task(dataset: str | ResolvedTaskSpec) -> ResolvedTaskSpec:
    if not isinstance(dataset, str):
        return dataset
    from judgearena.tasks.registry import get_packaged_task

    resolved = get_packaged_task(dataset)
    if resolved is None:
        raise ValueError(f"Unsupported task dataset {dataset!r}.")
    return resolved


def load_instructions(
    dataset: str | ResolvedTaskSpec, n_instructions: int | None = None
) -> pd.DataFrame:
    """Load instructions by task ID or an already-resolved task definition."""
    resolved_task = _resolve_task(dataset)
    from judgearena import utils as judgearena_utils
    from judgearena.datasets.registry import resolve_dataset_adapter

    adapter = resolve_dataset_adapter(resolved_task.spec.dataset.adapter)
    df_instructions = adapter.load_instructions(
        resolved_task, judgearena_utils.data_root / "tables"
    )

    df_instructions = df_instructions.set_index("instruction_index").sort_index()
    logger.info("Loaded %d instructions for %s.", len(df_instructions), resolved_task.task)
    if n_instructions is None:
        n_instructions = len(df_instructions)
    return df_instructions.head(n_instructions)


def load_battles(dataset: str | ResolvedTaskSpec) -> pd.DataFrame:
    """Load human preference battles for a battle-backed task."""
    resolved_task = _resolve_task(dataset)
    from judgearena import utils as judgearena_utils
    from judgearena.datasets.registry import resolve_battle_dataset_adapter

    adapter = resolve_battle_dataset_adapter(resolved_task.spec.dataset.adapter)
    battles = adapter.load_battles(
        resolved_task, judgearena_utils.data_root / "tables"
    )
    logger.info("Loaded %d battles for %s.", len(battles), resolved_task.task)
    return battles


if __name__ == "__main__":
    instructions = load_instructions(dataset="alpaca-eval")
    print(instructions)
