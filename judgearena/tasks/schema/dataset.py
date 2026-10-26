"""Common dataset contract for declarative tasks."""

from typing import Annotated, Any, Literal

from pydantic import Field

from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.tasks.schema.sources import SourceSpec


class DatasetFields(StrictFrozenModel):
    """Name source fields for adapters that support declarative instruction mapping.

    ``id`` and ``instruction`` name the upstream ID and prompt fields;
    ``category`` optionally names a source grouping field. These values are
    source names, unlike the source-to-canonical dictionaries on ``DatasetSpec``.
    The table adapter's ``id_strategy="position"`` replaces source IDs with row
    positions and remaps output IDs; ``"source"`` preserves the source IDs.
    Other adapters interpret the fields according to their source format.
    """

    id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    category: str | None = None
    id_strategy: Literal["source", "position"] = "source"


class DatasetSpec(StrictFrozenModel):
    """Declare how an adapter obtains and normalizes a dataset.

    ``adapter`` selects a registered dataset adapter; ``format`` is an optional
    adapter-specific source format. ``sources`` maps adapter-defined names such
    as ``"examples"`` to source declarations. Optional ``fields`` names instruction
    fields; ``options`` holds adapter-specific settings such as a local file path.
    Each adapter validates the settings it needs.

    ``output_columns`` and ``battle_columns`` map source column names to canonical
    names, e.g. ``{"answer": "output"}`` or ``{"left_model": "model_a"}``.
    ``outcome_preferences`` maps source outcome labels to numbers in [0, 1],
    e.g. ``{"left": 0, "draw": 0.5, "right": 1}``, using the A/B slot convention
    documented by ``PreferenceData``.

    Adapters apply the settings they support; these declarations do not perform
    renaming themselves. The returned ``CompletionData``/``PreferenceData`` owns
    the normalized data contract and pipeline compatibility checks use that data.
    """

    adapter: str = Field(min_length=1)
    # Adapter-specific source format, independent of the runner.
    format: str | None = None
    sources: dict[str, SourceSpec] = Field(default_factory=dict)
    fields: DatasetFields | None = None
    options: dict[str, Any] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    output_columns: dict[str, str] = Field(default_factory=dict)
    battle_columns: dict[str, str] = Field(default_factory=dict)
    outcome_preferences: dict[str, Annotated[float, Field(ge=0, le=1)]] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
