"""Baseline-selection policies shared by pairwise protocols."""

from typing import Annotated, Literal

from pydantic import Field, field_validator

from judgearena.tasks.schema.base import StrictFrozenModel


class NoBaseline(StrictFrozenModel):
    strategy: Literal["none"]


class RuntimeRequiredBaseline(StrictFrozenModel):
    strategy: Literal["runtime_required"]


class TaskDefaultBaseline(StrictFrozenModel):
    strategy: Literal["task_default"]
    reference_id: str = Field(min_length=1)


class AllReferencesBaseline(StrictFrozenModel):
    """Compare every selected instruction against each listed reference model."""

    strategy: Literal["all_references"]
    references: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1)

    @field_validator("references")
    @classmethod
    def _unique_references(cls, references):
        if len(set(references)) != len(references):
            raise ValueError("references must not contain duplicates")
        return references


class CategoryDefaultsBaseline(StrictFrozenModel):
    strategy: Literal["category_defaults"]
    category_field: str = Field(min_length=1)
    references: dict[str, str] = Field(min_length=1)


class OfficialOutputsBaseline(StrictFrozenModel):
    strategy: Literal["official_outputs"]
    source: str = Field(min_length=1)


BaselineSpec = Annotated[
    NoBaseline
    | RuntimeRequiredBaseline
    | TaskDefaultBaseline
    | AllReferencesBaseline
    | CategoryDefaultsBaseline
    | OfficialOutputsBaseline,
    Field(discriminator="strategy"),
]
