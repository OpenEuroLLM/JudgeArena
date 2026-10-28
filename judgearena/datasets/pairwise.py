"""Canonical data contract for registered single-turn pairwise tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from judgearena.log import get_logger
from judgearena.tasks.schema import ResolvedTaskSpec

logger = get_logger(__name__)


@dataclass(frozen=True)
class PairwiseTaskData:
    """Normalized instructions and optional pre-generated model outputs.

    ``model_outputs`` is ``None`` when the dataset ships no pre-generated
    completions; the runner generates whatever it cannot find here.
    """

    instructions: pd.DataFrame
    model_outputs: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.model_outputs is None:
            return
        required = {"instruction_index", "model", "output"}
        missing = sorted(required - set(self.model_outputs.columns))
        if missing:
            raise ValueError(
                f"Pairwise model outputs are missing canonical columns: {missing}."
            )

    def model_completion(
        self,
        model: str,
        *,
        instruction_ids: pd.Index | None = None,
    ) -> pd.Series | None:
        """Return model completions aligned to the requested instruction IDs."""
        if self.model_outputs is None:
            return None

        outputs = self.model_outputs.loc[
            self.model_outputs["model"] == model,
            ["instruction_index", "output"],
        ].copy()
        if outputs.empty:
            return None

        outputs["output"] = outputs["output"].fillna("")
        completions = (
            outputs.drop_duplicates("instruction_index", keep="last")
            .set_index("instruction_index")["output"]
            .sort_index()
        )
        target_index = (
            self.instructions.index
            if instruction_ids is None
            else pd.Index(instruction_ids, name=self.instructions.index.name)
        )
        logger.info("Found pre-existing completions for model %r.", model)
        return completions.loc[target_index].rename("completion")


def load_pairwise_task_data(
    task: ResolvedTaskSpec,
    *,
    n_instructions: int | None = None,
    local_tables_path: Path | None = None,
):
    """Load canonical data through the task's dataset adapter."""
    from dataclasses import replace

    from judgearena.datasets import load_task_data, resolve_dataset_filters

    data = load_task_data(task, local_tables_path=local_tables_path)
    data = data.select(resolve_dataset_filters(task))
    if n_instructions is not None:
        ids = data.instructions.head(n_instructions).index
        from judgearena.datasets.data import PreferenceData

        selected_battles = (
            {"battles": data.battles.loc[data.battles.instruction_index.isin(ids)]}
            if isinstance(data, PreferenceData)
            else {}
        )
        data = replace(
            data,
            **selected_battles,
            instructions=data.instructions.loc[ids],
            model_outputs=None
            if data.model_outputs is None
            else data.model_outputs.loc[data.model_outputs.instruction_index.isin(ids)],
        )
    return data
