"""Typed contract for task YAML; this module performs no loading or execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictFrozenModel(BaseModel):
    """Immutable schema node that rejects unknown YAML fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class HuggingFaceDatasetSource(_StrictFrozenModel):
    type: Literal["huggingface_dataset"]
    repo_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    config: str | None = None
    split: str | None = None
    allow_patterns: tuple[str, ...] = ()


class HuggingFaceSpaceSource(_StrictFrozenModel):
    type: Literal["huggingface_space"]
    repo_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    allow_patterns: tuple[str, ...] = ()


class GitRawSource(_StrictFrozenModel):
    type: Literal["git_raw"]
    repository: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    path: str = Field(min_length=1)


class LocalSource(_StrictFrozenModel):
    type: Literal["local"]
    path: str = Field(min_length=1)
    format: Literal["csv", "json", "jsonl", "parquet"]
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


SourceSpec = Annotated[
    HuggingFaceDatasetSource | HuggingFaceSpaceSource | GitRawSource | LocalSource,
    Field(discriminator="type"),
]


class DatasetFields(_StrictFrozenModel):
    id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    category: str | None = None


class DatasetSpec(_StrictFrozenModel):
    """Dataset sources, loader adapter, and canonical field mapping."""

    adapter: str = Field(min_length=1)
    sources: dict[str, SourceSpec] = Field(min_length=1)
    fields: DatasetFields


class SingleTurnGeneration(_StrictFrozenModel):
    mode: Literal["single_turn_chat"]


class NoBaseline(_StrictFrozenModel):
    strategy: Literal["none"]


class RuntimeRequiredBaseline(_StrictFrozenModel):
    strategy: Literal["runtime_required"]


class TaskDefaultBaseline(_StrictFrozenModel):
    strategy: Literal["task_default"]
    reference_id: str = Field(min_length=1)
    allow_runtime_override: bool = True


class CategoryDefaultsBaseline(_StrictFrozenModel):
    strategy: Literal["category_defaults"]
    category_field: str = Field(min_length=1)
    references: dict[str, str] = Field(min_length=1)
    allow_runtime_override: bool = True


class OfficialOutputsBaseline(_StrictFrozenModel):
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

SwapMode = Literal["fixed", "both"]


class PairwiseJudgeSpec(_StrictFrozenModel):
    default_prompt: str = Field(min_length=1)
    parser: str = Field(min_length=1)
    default_swap_mode: SwapMode = "fixed"
    allowed_swap_modes: tuple[SwapMode, ...] = ("fixed", "both")
    default_temperature: float | None = None

    @model_validator(mode="after")
    def _default_must_be_allowed(self) -> PairwiseJudgeSpec:
        if not self.allowed_swap_modes:
            raise ValueError("allowed_swap_modes must not be empty")
        if self.default_swap_mode not in self.allowed_swap_modes:
            raise ValueError("default_swap_mode must be present in allowed_swap_modes")
        if len(set(self.allowed_swap_modes)) != len(self.allowed_swap_modes):
            raise ValueError("allowed_swap_modes must not contain duplicates")
        return self


class ScoringSpec(_StrictFrozenModel):
    adapter: str = Field(min_length=1)
    primary_metric: str = Field(min_length=1)
    higher_is_better: bool


class PairwiseProtocol(_StrictFrozenModel):
    """Task-owned generation, baseline, judging, and scoring behavior."""

    runner: Literal["pairwise"]
    generation: SingleTurnGeneration
    baseline: BaselineSpec
    judge: PairwiseJudgeSpec
    scoring: ScoringSpec


class TaskMetadata(_StrictFrozenModel):
    reference_implementation: str | None = None
    paper: str | None = None


class SuffixVariants(_StrictFrozenModel):
    """Validated suffixes selecting views of one task definition."""

    selector: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
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


class TaskSpec(_StrictFrozenModel):
    """Complete validated definition of one registered task."""

    schema_version: Literal[1]
    task: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    task_version: int = Field(ge=1)
    description: str = Field(min_length=1)
    tags: tuple[str, ...] = ()
    dataset: DatasetSpec
    protocol: PairwiseProtocol
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


@dataclass(frozen=True)
class ResourceDigest:
    """Hash of one YAML resource used to construct a task."""

    path: str
    sha256: str


@dataclass(frozen=True)
class TaskProvenance:
    """Source paths and hashes needed to identify a resolved definition."""

    source_path: str
    source_sha256: str
    resolved_sha256: str
    resources: tuple[ResourceDigest, ...]


@dataclass(frozen=True)
class TaskSelection:
    """Runtime selector resolved from a task-family suffix."""

    selector: str
    name: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedTaskSpec:
    """Validated task plus the provenance of its resolved YAML."""

    spec: TaskSpec
    provenance: TaskProvenance
    invocation_task: str | None = None
    selection: TaskSelection | None = None

    @property
    def task(self) -> str:
        return self.invocation_task or self.spec.task

    @property
    def definition_task(self) -> str:
        """Task ID written in the source YAML, before suffix selection."""
        return self.spec.task

    def model_dump(self) -> dict[str, object]:
        """Return the normalized task definition without provenance."""
        return self.spec.model_dump(mode="json")
