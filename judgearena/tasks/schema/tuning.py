"""Task-owned defaults for judge configuration search."""

from typing import Any, Literal

from pydantic import Field

from judgearena.tasks.schema.base import StrictFrozenModel


class TuneJudgeProtocol(StrictFrozenModel):
    runner: Literal["tune_judge"]
    default_meta_eval_task: str
    default_objectives: list[Literal["agreement", "cost"]]
    default_fidelity: dict[str, dict[str, int]]
    default_optimizer: dict[str, Any] = Field(min_length=1)
