"""Pinned external resource schemas used by task datasets."""

from typing import Annotated, Literal

from pydantic import Field

from judgearena.tasks.schema.base import StrictFrozenModel


class HuggingFaceDatasetSource(StrictFrozenModel):
    type: Literal["huggingface_dataset"]
    repo_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    config: str | None = None
    split: str | None = None
    allow_patterns: tuple[str, ...] = ()


class HuggingFaceSpaceSource(StrictFrozenModel):
    type: Literal["huggingface_space"]
    repo_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    allow_patterns: tuple[str, ...] = ()


class GitRawSource(StrictFrozenModel):
    type: Literal["git_raw"]
    repository: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    path: str = Field(min_length=1)


class LocalSource(StrictFrozenModel):
    type: Literal["local"]
    path: str = Field(min_length=1)
    format: Literal["csv", "json", "jsonl", "parquet"]
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


SourceSpec = Annotated[
    HuggingFaceDatasetSource | HuggingFaceSpaceSource | GitRawSource | LocalSource,
    Field(discriminator="type"),
]
