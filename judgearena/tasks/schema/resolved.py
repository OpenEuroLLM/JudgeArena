"""Runtime records produced after loading and resolving task YAML."""

from __future__ import annotations

from dataclasses import dataclass

from judgearena.tasks.schema.task import TaskSpec


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
    prompt_text: str | None = None

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
