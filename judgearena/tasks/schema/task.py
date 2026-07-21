"""Top-level task definition and cross-section validation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.baselines import (
    CategoryDefaultsBaseline,
    OfficialOutputsBaseline,
)
from judgearena.tasks.schema.dataset import DatasetSpec
from judgearena.tasks.schema.elo import EloProtocol
from judgearena.tasks.schema.mt_bench import MTBenchProtocol
from judgearena.tasks.schema.pairwise import PairwiseProtocol
from judgearena.tasks.schema.wildbench import WildBenchProtocol

ProtocolSpec = Annotated[
    PairwiseProtocol | MTBenchProtocol | WildBenchProtocol | EloProtocol,
    Field(discriminator="runner"),
]


class TaskMetadata(StrictFrozenModel):
    reference_implementation: str | None = None
    paper: str | None = None


class SuffixVariants(StrictFrozenModel):
    """Validated suffixes selecting views of one task definition."""

    selector: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    separator: Literal["-"] = "-"
    values: tuple[str, ...] = Field(min_length=1)
    groups: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_variants(self) -> SuffixVariants:
        if any(not value for value in self.values):
            raise ValueError("variant values must not be empty")
        if any(not group for group in self.groups):
            raise ValueError("variant group names must not be empty")
        if len(set(self.values)) != len(self.values):
            raise ValueError("variant values must not contain duplicates")
        overlap = set(self.values) & set(self.groups)
        if overlap:
            raise ValueError(f"variant values and groups overlap: {sorted(overlap)}")
        known = set(self.values)
        for group, members in self.groups.items():
            if not members:
                raise ValueError(f"variant group {group!r} must not be empty")
            if len(set(members)) != len(members):
                raise ValueError(f"variant group {group!r} must not contain duplicates")
            unknown = sorted(set(members) - known)
            if unknown:
                raise ValueError(
                    f"variant group {group!r} references unknown values: {unknown}"
                )
        return self


class TaskSpec(StrictFrozenModel):
    """Complete validated definition of one registered task."""

    schema_version: Literal[1]
    task: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    task_version: int = Field(ge=1)
    description: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    dataset: DatasetSpec
    protocol: ProtocolSpec
    variants: SuffixVariants | None = None
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)

    @model_validator(mode="after")
    def _validate_task(self) -> TaskSpec:
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("tags must not contain duplicates")
        source_names = set(self.dataset.sources)
        baseline = self.protocol.baseline
        if (
            isinstance(baseline, OfficialOutputsBaseline)
            and baseline.source not in source_names
        ):
            raise ValueError(
                f"official baseline source {baseline.source!r} is not declared "
                "in dataset.sources"
            )
        if isinstance(baseline, CategoryDefaultsBaseline) and (
            self.dataset.fields.category != baseline.category_field
        ):
            raise ValueError(
                "category-default baseline must use dataset.fields.category"
            )
        return self
