"""Registry connecting task dataset-adapter IDs to implementations."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from judgearena.tasks.schema import DatasetFields, ResolvedTaskSpec

if TYPE_CHECKING:
    from judgearena.datasets.data import CompletionData


def source_cache_dir(task: ResolvedTaskSpec, local_tables_path: Path) -> Path:
    """Keep downloaded and normalized data specific to its declared inputs."""
    dataset = json.dumps(task.spec.dataset.model_dump(mode="json"), sort_keys=True)
    digest = hashlib.sha256(dataset.encode()).hexdigest()[:16]
    return local_tables_path / "_sources" / task.definition_task / digest


@dataclass
class DatasetAdapter(ABC):
    """Load a source into the dataset contracts consumed by pipelines.

    ``task`` holds the resolved dataset declaration and optional selected variant.
    ``cache_dir`` is the caller's local table/cache root; each adapter owns its
    layout and may also use a source library's cache. It is not necessarily the
    directory containing raw source files.

    Implement ``load_data()`` and decorate the class with
    ``@register_dataset("my_dataset")``. Interpret source formats and column
    mappings here, returning ``CompletionData`` or ``PreferenceData`` with useful
    metadata preserved. Override ``download()`` when sources need downloading.
    The returned data determines pipeline compatibility.
    """

    task: ResolvedTaskSpec
    cache_dir: Path

    @property
    def fields(self) -> DatasetFields:
        """Require instruction-field mappings when an adapter uses them."""
        fields = self.task.spec.dataset.fields
        if fields is None:
            raise ValueError(
                f"Dataset adapter {self.task.spec.dataset.adapter!r} requires dataset.fields."
            )
        return fields

    def download(self) -> None:
        """Download missing sources when needed; the default does nothing."""
        return None

    @abstractmethod
    def load_data(self) -> CompletionData:
        """Return validated data, obtaining any missing sources needed to load it.

        Callers need not call ``download()`` first. Optional saved model answers
        may be supplied eagerly or through ``CompletionData.model_output_loader``.
        """


_DATASETS: dict[str, type[DatasetAdapter]] = {}


def register_dataset[T: type[DatasetAdapter]](name: str) -> Callable[[T], T]:
    """Register an adapter class under the ID used by ``dataset.adapter``.

    Use as ``@register_dataset("my_dataset")`` or call with an existing class.
    Duplicate names raise ``ValueError``. Registration lasts for this Python
    process; import custom adapters before loading tasks or creating an
    ``AdapterCatalog``, which snapshots the available names.
    """

    def register(adapter: T) -> T:
        if name in _DATASETS:
            raise ValueError(f"Dataset adapter {name!r} is already registered.")
        _DATASETS[name] = adapter
        return adapter

    return register


def dataset_names() -> frozenset[str]:
    return frozenset(_DATASETS)


def resolve_dataset_adapter(task: ResolvedTaskSpec, cache_dir: Path) -> DatasetAdapter:
    return _DATASETS[task.spec.dataset.adapter](task, cache_dir)


# Load built-ins after defining the API so decorators can register their classes.
from judgearena.datasets import (  # noqa: E402, F401
    arena_battles,
    arena_hard,
    fluency,
    judgearena_tables,
    m_arenahard,
    mt_bench,
    mt_bench_101,
)
