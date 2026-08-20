"""Common dataset contract for declarative tasks."""

from pydantic import Field

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.sources import SourceSpec


class DatasetFields(StrictFrozenModel):
    """Map upstream columns to JudgeArena's canonical task fields."""

    id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    category: str | None = None


class DatasetSpec(StrictFrozenModel):
    """Dataset sources, loader adapter, and canonical field mapping."""

    adapter: str = Field(min_length=1)
    sources: dict[str, SourceSpec] = Field(min_length=1)
    fields: DatasetFields
