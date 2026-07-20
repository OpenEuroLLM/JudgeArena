"""Shared behavior for immutable task schema models."""

from pydantic import BaseModel, ConfigDict


class StrictFrozenModel(BaseModel):
    """Immutable schema node that rejects unknown YAML fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)
