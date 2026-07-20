"""Canonical data contract for registered single-turn pairwise tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from judgearena.datasets.registry import resolve_dataset_adapter
from judgearena.log import get_logger
from judgearena.tasks.schema import PairwiseProtocol, ResolvedTaskSpec
from judgearena.utils import data_root

logger = get_logger(__name__)


@dataclass(frozen=True)
class PairwiseTaskData:
    """Normalized instructions and optional pre-generated model outputs."""

    instructions: pd.DataFrame
    model_outputs: pd.DataFrame | None = None

    def completions_for(self, model: str) -> pd.Series | None:
        """Return model completions aligned to the instruction index."""
        if self.model_outputs is None:
            return None

        required = {"instruction_index", "model", "output"}
        missing = sorted(required - set(self.model_outputs.columns))
        if missing:
            raise ValueError(
                f"Pairwise model outputs are missing canonical columns: {missing}."
            )

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
        logger.info("Found pre-existing completions for model %r.", model)
        return completions.loc[self.instructions.index].rename("completion")


def load_pairwise_task_data(
    task: ResolvedTaskSpec,
    *,
    n_instructions: int | None = None,
    local_tables_path: Path | None = None,
) -> PairwiseTaskData:
    """Load one registered task through its declared dataset adapter."""
    if not isinstance(task.spec.protocol, PairwiseProtocol):
        raise ValueError(f"Task {task.task!r} does not use the pairwise protocol.")

    tables_path = local_tables_path or data_root / "tables"
    adapter = resolve_dataset_adapter(task.spec.dataset.adapter)
    instructions = adapter.load_instructions(task, tables_path)
    if "instruction_index" in instructions.columns:
        instructions = instructions.set_index("instruction_index")
    if instructions.index.name != "instruction_index":
        raise ValueError(
            f"Dataset adapter {adapter.name!r} must provide 'instruction_index'."
        )
    if "instruction" not in instructions.columns:
        raise ValueError(
            f"Dataset adapter {adapter.name!r} must provide 'instruction'."
        )
    if instructions.index.has_duplicates:
        raise ValueError(
            f"Dataset adapter {adapter.name!r} returned duplicate instruction IDs."
        )

    instructions = instructions.sort_index()
    if n_instructions is not None:
        instructions = instructions.head(n_instructions)
    logger.info("Loaded %d instructions for %s.", len(instructions), task.task)

    return PairwiseTaskData(
        instructions=instructions,
        model_outputs=adapter.load_model_outputs(task, tables_path),
    )
