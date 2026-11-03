from collections.abc import Collection, Mapping
from pathlib import Path

import pandas as pd

from judgearena.datasets.data import CompletionData, PreferenceData
from judgearena.tasks.schema import ResolvedTaskSpec


def _resolve_task(dataset: str | ResolvedTaskSpec) -> ResolvedTaskSpec:
    if not isinstance(dataset, str):
        return dataset
    from judgearena.tasks.registry import get_packaged_task

    resolved = get_packaged_task(dataset)
    if resolved is None:
        raise ValueError(f"Unsupported task dataset {dataset!r}.")
    return resolved


def load_task_data(
    dataset: str | ResolvedTaskSpec, *, local_tables_path: Path | None = None
) -> CompletionData:
    """Load a dataset through its adapter and validate the returned capability."""
    from judgearena import utils as judgearena_utils
    from judgearena.datasets.registry import resolve_dataset_adapter

    task = _resolve_task(dataset)
    data = resolve_dataset_adapter(
        task, local_tables_path or judgearena_utils.data_root / "tables"
    ).load_data()
    if not isinstance(data, CompletionData):
        raise TypeError(
            f"Dataset adapter {task.spec.dataset.adapter!r} must return CompletionData."
        )
    return data


def load_instructions(
    dataset: str | ResolvedTaskSpec, n_instructions: int | None = None
) -> pd.DataFrame:
    """Load the instruction view while retaining benchmark-specific metadata."""
    task = _resolve_task(dataset)
    instructions = (
        load_task_data(task).select(resolve_dataset_filters(task)).instructions
    )
    return instructions if n_instructions is None else instructions.head(n_instructions)


def load_battles(dataset: str | ResolvedTaskSpec) -> pd.DataFrame:
    """Load the preference view, retaining source columns for specialized runners."""
    task = _resolve_task(dataset)
    data = load_task_data(task).select(resolve_dataset_filters(task))
    if not isinstance(data, PreferenceData):
        raise ValueError("Battle evaluation requires PreferenceData.")
    return data.battles


def resolve_dataset_filters(
    task: ResolvedTaskSpec, *requested: Mapping[str, Collection[object]]
) -> dict[str, list[object]]:
    """Intersect task variants and runtime filters on their declared metadata fields."""
    filters = (
        {task.selection.selector: list(task.selection.values)}
        if task.selection is not None
        else {}
    )
    for selection in requested:
        for column, values in selection.items():
            values = list(values)
            if column in filters:
                values = [value for value in filters[column] if value in values]
                if not values:
                    raise ValueError(f"Dataset filters have no overlap for {column!r}.")
            filters[column] = values
    return filters
