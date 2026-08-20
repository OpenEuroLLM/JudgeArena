"""Baseline-selection policies shared by pairwise protocols."""

from typing import Annotated, Literal

from pydantic import Field

from judgearena.tasks.schema.base import StrictFrozenModel


class NoBaseline(StrictFrozenModel):
    strategy: Literal["none"]


class RuntimeRequiredBaseline(StrictFrozenModel):
    strategy: Literal["runtime_required"]


class TaskDefaultBaseline(StrictFrozenModel):
    strategy: Literal["task_default"]
    reference_id: str = Field(min_length=1)
    allow_runtime_override: bool = True


class CategoryDefaultsBaseline(StrictFrozenModel):
    strategy: Literal["category_defaults"]
    category_field: str = Field(min_length=1)
    references: dict[str, str] = Field(min_length=1)
    allow_runtime_override: bool = True


class OfficialOutputsBaseline(StrictFrozenModel):
    strategy: Literal["official_outputs"]
    source: str = Field(min_length=1)


BaselineSpec = Annotated[
    NoBaseline
    | RuntimeRequiredBaseline
    | TaskDefaultBaseline
    | CategoryDefaultsBaseline
    | OfficialOutputsBaseline,
    Field(discriminator="strategy"),
]
