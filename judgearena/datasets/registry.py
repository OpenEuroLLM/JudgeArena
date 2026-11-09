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
from judgearena.utils.extensions import (
    external_component_reference,
    load_python_object,
    load_task_components,
    python_reference,
)

if TYPE_CHECKING:
    from judgearena.datasets.data import CompletionData


def source_cache_dir(task: ResolvedTaskSpec, local_tables_path: Path) -> Path:
    """Keep downloaded and normalized data specific to its declared inputs."""
    declaration = task.spec.dataset.model_dump(mode="json")
    adapter = declaration["adapter"]
    reference = python_reference(adapter)
    if reference is None:
        reference = python_reference(
            external_component_reference(dataset_adapter_type(adapter))
        )
    if reference is not None:
        path, symbol = reference
        declaration["adapter"] = (
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}:{symbol}"
        )
    dataset = json.dumps(declaration, sort_keys=True)
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
    ``@register_dataset("my_dataset")`` or reference it as ``./dataset.py:Adapter``
    from task YAML. Interpret source formats and column
    mappings here, returning ``CompletionData`` or ``PreferenceData`` with useful
    metadata preserved. Override ``download()`` when sources need downloading.
    The returned data determines pipeline compatibility.
    """

    task: ResolvedTaskSpec
    cache_dir: Path

    @property
    def source_dir(self) -> Path:
        """Local files isolated by the task's declared dataset sources and options."""
        return source_cache_dir(self.task, self.cache_dir)

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
    load_task_components()
    return frozenset(_DATASETS)


def resolve_dataset_adapter(task: ResolvedTaskSpec, cache_dir: Path) -> DatasetAdapter:
    return dataset_adapter_type(task.spec.dataset.adapter)(task, cache_dir)


def dataset_adapter_type(name: str) -> type[DatasetAdapter]:
    reference = python_reference(name)
    if reference is not None:
        adapter = load_python_object(*reference)
        if not isinstance(adapter, type) or not issubclass(adapter, DatasetAdapter):
            raise ValueError(f"{name!r} must reference a DatasetAdapter class.")
        return adapter
    load_task_components()
    try:
        return _DATASETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset adapter {name!r}.") from exc


# Shared adapters are used by multiple task families.
from judgearena.datasets import arena_battles, judgearena_tables  # noqa: E402, F401
