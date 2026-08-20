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

    def load_model_completions(
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
        missing = target_index[~target_index.isin(completions.index)]
        if len(missing):
            preview = missing[:5].tolist()
            suffix = "..." if len(missing) > len(preview) else ""
            raise ValueError(
                f"Pre-existing completions for model {model!r} are missing "
                f"{len(missing)} required instruction(s): {preview}{suffix}"
            )
        logger.info("Found pre-existing completions for model %r.", model)
        return completions.loc[target_index].rename("completion")


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
    adapter_id = task.spec.dataset.adapter
    adapter = resolve_dataset_adapter(adapter_id)
    instructions = adapter.load_instructions(task, tables_path)
    if "instruction_index" in instructions.columns:
        instructions = instructions.set_index("instruction_index")
    if instructions.index.name != "instruction_index":
        raise ValueError(
            f"Dataset adapter {adapter_id!r} must provide 'instruction_index'."
        )
    if "instruction" not in instructions.columns:
        raise ValueError(f"Dataset adapter {adapter_id!r} must provide 'instruction'.")
    if instructions.index.has_duplicates:
        raise ValueError(
            f"Dataset adapter {adapter_id!r} returned duplicate instruction IDs."
        )

    instructions = instructions.sort_index()
    if n_instructions is not None:
        instructions = instructions.head(n_instructions)
    logger.info("Loaded %d instructions for %s.", len(instructions), task.task)

    return PairwiseTaskData(
        instructions=instructions,
        model_outputs=adapter.load_model_outputs(task, tables_path),
    )
