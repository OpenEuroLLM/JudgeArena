"""Load packaged task definitions and validate their referenced component IDs."""

from __future__ import annotations

import hashlib
import json
import posixpath
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import ValidationError

from judgearena.benchmarks.scoring import available_metrics, build_metric
from judgearena.log import get_logger
from judgearena.prompts.registry import JUDGE_PROMPT_PRESETS
from judgearena.tasks.schema import (
    EloProtocol,
    MTBenchProtocol,
    ResolvedTaskSpec,
    ResourceDigest,
    TaskProvenance,
    TaskSelection,
    TaskSpec,
)

logger = get_logger(__name__)


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


def _normalize_root_path(relative_path: str) -> str:
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


def _resource(root: Traversable, relative_path: str) -> Traversable:
    normalized = _normalize_root_path(relative_path)
    resource: Traversable = root
    for part in PurePosixPath(normalized).parts:
        resource = resource.joinpath(part)
    if isinstance(root, Path) and isinstance(resource, Path):
        root_path = root.resolve()
        resource_path = resource.resolve()
        if not resource_path.is_relative_to(root_path):
            raise TaskDefinitionError(
                f"{relative_path}: path escapes task definitions root"
            )
    return resource


def _discover(root: Traversable) -> tuple[str, ...]:
    """Public task YAMLs at exactly ``<family>/<task>.yaml``.

    ``_*.yaml`` files are private bases for ``extends`` and are not runnable.
    """
    if not root.is_dir():
        raise TaskDefinitionError("task definitions root is not a directory")
    discovered: list[str] = []
    for family in sorted(root.iterdir(), key=lambda entry: entry.name):
        if not family.is_dir():
            if family.name.endswith((".yaml", ".yml")):
                logger.warning(
                    "Ignoring %s: task YAML must live in a <family>/ subfolder.",
                    family.name,
                )
            continue
        for entry in sorted(family.iterdir(), key=lambda item: item.name):
            if (
                entry.is_file()
                and entry.name.endswith((".yaml", ".yml"))
                and not entry.name.startswith("_")
            ):
                discovered.append(f"{family.name}/{entry.name}")
    return tuple(discovered)


def _resolve(
    root: Traversable, relative_path: str, *, chain: tuple[str, ...]
) -> tuple[dict[str, Any], tuple[ResourceDigest, ...]]:
    normalized_path = _normalize_root_path(relative_path)
    if normalized_path in chain:
        cycle = " -> ".join((*chain, normalized_path))
        raise TaskDefinitionError(f"task inheritance cycle: {cycle}")

    resource = _resource(root, normalized_path)
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
    base_path = _resolve_extends(normalized_path, extends)
    base, base_resources = _resolve(root, base_path, chain=(*chain, normalized_path))
    return _merge_mapping(base, data), (*base_resources, digest)


def _resolve_extends(child_path: str, extends: str) -> str:
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
    return _normalize_root_path(joined)


def _load_task(root: Traversable, relative_path: str) -> ResolvedTaskSpec:
    """Resolve, validate, and fingerprint one public task definition."""
    relative_path = _normalize_root_path(relative_path)
    if PurePosixPath(relative_path).name.startswith("_"):
        raise TaskDefinitionError(
            f"{relative_path}: private base files are not runnable tasks"
        )
    resolved, resources = _resolve(root, relative_path, chain=())
    if "task" not in resolved:
        raise TaskDefinitionError(f"{relative_path}: public task must define task")
    try:
        spec = TaskSpec.model_validate(resolved)
    except ValidationError as exc:
        raise TaskDefinitionError(f"{relative_path}: {exc}") from exc
    normalized = spec.model_dump(mode="json")
    child_digest = next(digest for digest in resources if digest.path == relative_path)
    return ResolvedTaskSpec(
        spec=spec,
        provenance=TaskProvenance(
            source_path=relative_path,
            source_sha256=child_digest.sha256,
            resolved_sha256=_canonical_sha256(normalized),
            resources=resources,
        ),
    )


@dataclass(frozen=True)
class AdapterCatalog:
    """Component IDs that task YAML files may reference."""

    runners: frozenset[str] = frozenset({"elo", "mt_bench", "pairwise"})
    # Keep in sync with judgearena.datasets.registry; importing it here
    # would create a cycle.
    instruction_datasets: frozenset[str] = frozenset(
        {"arena_hard", "fluency", "judgearena_tables", "m_arena_hard", "mt_bench"}
    )
    battle_datasets: frozenset[str] = frozenset({"arena_battles"})
    prompts: frozenset[str] = frozenset(JUDGE_PROMPT_PRESETS)
    metrics: frozenset[str] = frozenset(available_metrics())


def load_tasks(
    definitions_root: Traversable | None = None,
    *,
    adapters: AdapterCatalog | None = None,
) -> dict[str, ResolvedTaskSpec]:
    """Discover, validate, and return all packaged tasks keyed by task ID.

    With no arguments this reads JudgeArena's installed definitions and caches
    the result. An explicit ``definitions_root`` (used by tests) is never cached.
    """
    if definitions_root is None and adapters is None:
        return _load_packaged_tasks()
    root = definitions_root or files("judgearena.tasks").joinpath("definitions")
    return _discover_tasks(root, adapters or AdapterCatalog())


@cache
def _load_packaged_tasks() -> dict[str, ResolvedTaskSpec]:
    root = files("judgearena.tasks").joinpath("definitions")
    return _discover_tasks(root, AdapterCatalog())


def _discover_tasks(
    definitions_root: Traversable, adapters: AdapterCatalog
) -> dict[str, ResolvedTaskSpec]:
    tasks: dict[str, ResolvedTaskSpec] = {}
    for relative_path in _discover(definitions_root):
        resolved = _load_task(definitions_root, relative_path)
        if resolved.task in tasks:
            other = tasks[resolved.task].provenance.source_path
            raise TaskDefinitionError(
                f"Duplicate task ID {resolved.task!r} in {other} and {relative_path}"
            )
        _validate_adapter_ids(resolved, adapters)
        tasks[resolved.task] = resolved
    tasks = dict(sorted(tasks.items()))
    _validate_variant_ids(tasks)
    return tasks


def _validate_adapter_ids(resolved: ResolvedTaskSpec, adapters: AdapterCatalog) -> None:
    spec = resolved.spec
    is_elo = isinstance(spec.protocol, EloProtocol)
    dataset_names = (
        adapters.battle_datasets if is_elo else adapters.instruction_datasets
    )
    references = {
        "runner": (spec.protocol.runner, adapters.runners),
        "dataset adapter": (spec.dataset.adapter, dataset_names),
    }
    metric_names = adapters.metrics
    for request in spec.protocol.scoring.metrics:
        if request.metric not in metric_names:
            raise TaskDefinitionError(
                f"{resolved.provenance.source_path}: unknown metric {request.metric!r}"
            )
        try:
            build_metric(request.metric, request.parameters)
        except ValueError as exc:
            raise TaskDefinitionError(
                f"{resolved.provenance.source_path}: invalid metric "
                f"{request.metric!r}: {exc}"
            ) from exc
        if isinstance(spec.protocol, MTBenchProtocol):
            unsupported = sorted(set(request.group_by) - {"category", "turn"})
            if unsupported:
                raise TaskDefinitionError(
                    f"{resolved.provenance.source_path}: MT-Bench cannot group "
                    f"metrics by {unsupported}"
                )
    judge = spec.protocol.judge
    references["prompt"] = (judge.default_prompt_preset, adapters.prompts)
    for kind, (adapter_id, available) in references.items():
        if adapter_id not in available:
            raise TaskDefinitionError(
                f"{resolved.provenance.source_path}: unknown {kind} {adapter_id!r}"
            )
    category_prompts = getattr(spec.protocol.judge, "category_prompts", {})
    for category, preset in category_prompts.items():
        if preset not in adapters.prompts:
            raise TaskDefinitionError(
                f"{resolved.provenance.source_path}: unknown prompt {preset!r} "
                f"for category {category!r}"
            )


def _validate_variant_ids(tasks: dict[str, ResolvedTaskSpec]) -> None:
    """Reject a family-variant ID that collides with a real task or each other."""
    aliases: dict[str, str] = {}
    for task_id, resolved in tasks.items():
        variants = resolved.spec.variants
        if variants is None:
            continue
        for suffix in (*variants.values, *variants.groups):
            alias = f"{task_id}-{suffix}"
            if alias in tasks:
                raise TaskDefinitionError(
                    f"Variant {alias!r} from {task_id!r} collides with an "
                    "existing task of the same ID."
                )
            if alias in aliases:
                raise TaskDefinitionError(
                    f"Variant {alias!r} from {task_id!r} collides with a variant "
                    f"already generated by {aliases[alias]!r}."
                )
            aliases[alias] = task_id


def resolve_task(
    tasks: dict[str, ResolvedTaskSpec], task_id: str
) -> ResolvedTaskSpec | None:
    """Resolve ``task_id`` against ``tasks``, or ``None`` when it is unknown.

    A ``task_id`` matches either a definition directly, or a task family:
    ``family-v1-uk`` selects the ``uk`` view of the ``family-v1`` definition,
    where ``uk`` is a single value or a named group declared under ``variants``.
    """
    exact = tasks.get(task_id)
    if exact is not None:
        return exact

    # Longest definition name first, so a longer base wins over a shorter one
    # that is a prefix of it (e.g. ``a-b`` before ``a``).
    for base in sorted(tasks.values(), key=lambda t: len(t.task), reverse=True):
        variants = base.spec.variants
        if variants is None:
            continue
        prefix = f"{base.task}-"
        if not task_id.startswith(prefix):
            continue
        suffix = task_id[len(prefix) :]
        if suffix in variants.values:
            values: tuple[str, ...] = (suffix,)
        elif suffix in variants.groups:
            values = variants.groups[suffix]
        else:
            continue
        return ResolvedTaskSpec(
            spec=base.spec,
            provenance=base.provenance,
            invocation_task=task_id,
            selection=TaskSelection(
                selector=variants.selector,
                name=suffix,
                values=values,
            ),
        )
    return None


def get_packaged_task(task_id: str) -> ResolvedTaskSpec | None:
    """Look up a task from JudgeArena's installed YAML definitions.

    Family variants (e.g. ``family-v1-uk``) are resolved from their definition.
    """
    return resolve_task(load_tasks(), task_id)
