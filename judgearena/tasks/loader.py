"""Read task YAML, resolve private base files, and record stable hashes."""

from __future__ import annotations

import hashlib
import json
import posixpath
from copy import deepcopy
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import ValidationError

from judgearena.tasks.schema import (
    ResolvedTaskSpec,
    ResourceDigest,
    TaskProvenance,
    TaskSpec,
)


class TaskDefinitionError(ValueError):
    """A packaged task definition is malformed or unsafe."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""

    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(data: dict[str, object]) -> str:
    canonical = json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return _sha256(canonical)


def _strict_yaml_mapping(text: str, *, path: str) -> dict[str, Any]:
    try:
        data = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise TaskDefinitionError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise TaskDefinitionError(f"{path}: task YAML must contain a mapping")
    if any(not isinstance(key, str) for key in data):
        raise TaskDefinitionError(f"{path}: task YAML keys must be strings")
    return data


def _merge_mapping(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Apply the documented recursive-map/replace-list/null-delete merge."""
    merged = deepcopy(parent)
    for key, value in child.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class TaskLoader:
    """Load and normalize task YAML from a filesystem or installed package."""

    def __init__(self, root: Traversable):
        self.root = root

    def discover(self) -> tuple[str, ...]:
        """Return public task YAML paths; ``_*.yaml`` files are private bases."""
        discovered: list[str] = []

        def walk(directory: Traversable, prefix: PurePosixPath) -> None:
            for entry in sorted(directory.iterdir(), key=lambda item: item.name):
                relative = prefix / entry.name
                if entry.is_dir():
                    walk(entry, relative)
                elif (
                    entry.is_file()
                    and entry.name.endswith((".yaml", ".yml"))
                    and not entry.name.startswith("_")
                ):
                    discovered.append(relative.as_posix())

        if not self.root.is_dir():
            raise TaskDefinitionError("task definitions root is not a directory")
        walk(self.root, PurePosixPath())
        return tuple(discovered)

    def load(self, relative_path: str) -> ResolvedTaskSpec:
        """Resolve, validate, and fingerprint one public task definition."""
        relative_path = self._normalize_root_path(relative_path)
        if PurePosixPath(relative_path).name.startswith("_"):
            raise TaskDefinitionError(
                f"{relative_path}: private base files are not runnable tasks"
            )
        resolved, resources = self._resolve(relative_path, chain=())
        if "task" not in resolved:
            raise TaskDefinitionError(f"{relative_path}: public task must define task")
        try:
            spec = TaskSpec.model_validate(resolved)
        except ValidationError as exc:
            raise TaskDefinitionError(f"{relative_path}: {exc}") from exc
        normalized = spec.model_dump(mode="json")
        child_digest = next(
            digest for digest in resources if digest.path == relative_path
        )
        return ResolvedTaskSpec(
            spec=spec,
            provenance=TaskProvenance(
                source_path=relative_path,
                source_sha256=child_digest.sha256,
                resolved_sha256=_canonical_sha256(normalized),
                resources=resources,
            ),
        )

    def _resolve(
        self, relative_path: str, *, chain: tuple[str, ...]
    ) -> tuple[dict[str, Any], tuple[ResourceDigest, ...]]:
        normalized_path = self._normalize_root_path(relative_path)
        if normalized_path in chain:
            cycle = " -> ".join((*chain, normalized_path))
            raise TaskDefinitionError(f"task inheritance cycle: {cycle}")

        resource = self._resource(normalized_path)
        if not resource.is_file():
            raise TaskDefinitionError(f"{normalized_path}: task file does not exist")
        text = resource.read_text(encoding="utf-8")
        data = _strict_yaml_mapping(text, path=normalized_path)
        digest = ResourceDigest(normalized_path, _sha256(text))

        is_base = PurePosixPath(normalized_path).name.startswith("_")
        if is_base and "task" in data:
            raise TaskDefinitionError(
                f"{normalized_path}: private base files must not define task"
            )

        extends = data.pop("extends", None)
        if extends is None:
            return data, (digest,)
        if not isinstance(extends, str) or not extends:
            raise TaskDefinitionError(
                f"{normalized_path}: extends must be one relative YAML path"
            )
        base_path = self._resolve_extends(normalized_path, extends)
        base, base_resources = self._resolve(base_path, chain=(*chain, normalized_path))
        return _merge_mapping(base, data), (*base_resources, digest)

    def _resolve_extends(self, child_path: str, extends: str) -> str:
        requested = PurePosixPath(extends)
        if requested.is_absolute() or requested.suffix not in {".yaml", ".yml"}:
            raise TaskDefinitionError(
                f"{child_path}: extends must reference a relative YAML file"
            )
        if not requested.name.startswith("_"):
            raise TaskDefinitionError(
                f"{child_path}: extends may only reference a private _*.yaml base"
            )
        joined = posixpath.normpath(str(PurePosixPath(child_path).parent / requested))
        return self._normalize_root_path(joined)

    def _normalize_root_path(self, relative_path: str) -> str:
        pure = PurePosixPath(relative_path)
        normalized = posixpath.normpath(pure.as_posix())
        if (
            pure.is_absolute()
            or normalized in {"", ".", ".."}
            or normalized.startswith("../")
        ):
            raise TaskDefinitionError(
                f"{relative_path}: path escapes task definitions root"
            )
        return PurePosixPath(normalized).as_posix()

    def _resource(self, relative_path: str) -> Traversable:
        normalized = self._normalize_root_path(relative_path)
        resource: Traversable = self.root
        for part in PurePosixPath(normalized).parts:
            resource = resource.joinpath(part)
        if isinstance(self.root, Path) and isinstance(resource, Path):
            root_path = self.root.resolve()
            resource_path = resource.resolve()
            if not resource_path.is_relative_to(root_path):
                raise TaskDefinitionError(
                    f"{relative_path}: path escapes task definitions root"
                )
        return resource
