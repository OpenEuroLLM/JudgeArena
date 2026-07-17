"""Compact reproducibility manifests for JudgeArena runs."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.prompts.registry import ResolvedJudgePrompt

METADATA_FILENAME = "run-metadata.v2.json"
METADATA_SCHEMA_VERSION = "judgearena-run-metadata/v2"
_REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")


class _MetadataModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelIdentity(_MetadataModel):
    """Configured model fields, named exactly as they are in ``RunConfig``."""

    name: str
    baseline: str | None = None


class JudgeIdentity(_MetadataModel):
    model: str


class EloIdentity(_MetadataModel):
    arena: str
    baseline_model: str | None = None


class RunIdentity(_MetadataModel):
    """Small, human-readable identity for one evaluation run."""

    workflow: str
    task: str | None = None
    model: ModelIdentity | None = None
    judge: JudgeIdentity | None = None
    elo: EloIdentity | None = None

    @classmethod
    def from_config(cls, cfg: RunConfig, *, workflow: str) -> RunIdentity:
        """Extract config-owned identity fields without renaming them."""
        elo = None
        if cfg.elo is not None and cfg.elo.arena is not None:
            elo = EloIdentity(
                arena=cfg.elo.arena,
                baseline_model=cfg.elo.baseline_model,
            )
        return cls(
            workflow=workflow,
            task=cfg.task,
            model=ModelIdentity(
                name=cfg.model.name,
                baseline=cfg.model.baseline,
            ),
            judge=JudgeIdentity(model=cfg.judge.model),
            elo=elo,
        )


class ConfigurationMetadata(_MetadataModel):
    path: str
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> ConfigurationMetadata:
        config_path = Path(path)
        return cls(path=config_path.name, sha256=_hash_file_sha256(config_path))


class RunContext(_MetadataModel):
    """Reusable metadata context for one run."""

    identity: RunIdentity
    configuration: ConfigurationMetadata | None = None

    @classmethod
    def from_config(
        cls,
        cfg: RunConfig,
        *,
        workflow: str,
        configuration_path: str | Path,
    ) -> RunContext:
        return cls(
            identity=RunIdentity.from_config(cfg, workflow=workflow),
            configuration=ConfigurationMetadata.from_path(configuration_path),
        )


class ExecutionMetadata(_MetadataModel):
    entrypoint: str
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float = Field(ge=0)


class InputMetadata(_MetadataModel):
    dataset_revisions: dict[str, str | None] | None = None
    example_count: int | None = Field(default=None, ge=0)
    example_ids_sha256: str | None = None
    content_sha256: str | None = None
    judgment_count: int | None = Field(default=None, ge=0)
    calibration_example_count: int | None = Field(default=None, ge=0)
    calibration_judgment_count: int | None = Field(default=None, ge=0)

    @classmethod
    def capture(
        cls,
        *,
        dataset_revisions: dict[str, str | None] | None = None,
        example_ids: Sequence[Any] | None = None,
        content: Any = None,
        example_count: int | None = None,
        judgment_count: int | None = None,
        calibration_example_count: int | None = None,
        calibration_judgment_count: int | None = None,
    ) -> InputMetadata:
        """Fingerprint inputs without embedding their full payload."""
        normalized_ids = (
            _to_jsonable(list(example_ids)) if example_ids is not None else None
        )
        if normalized_ids is not None and not isinstance(normalized_ids, list):
            raise TypeError("example_ids must normalize to a list.")
        if example_count is None and normalized_ids is not None:
            example_count = len(normalized_ids)
        elif normalized_ids is not None and example_count != len(normalized_ids):
            raise ValueError(
                "example_count must equal the number of supplied example_ids."
            )
        distinct_judgment_count = (
            judgment_count if judgment_count != example_count else None
        )
        return cls(
            dataset_revisions=dataset_revisions or None,
            example_count=example_count,
            example_ids_sha256=_hash_normalized_set_sha256(normalized_ids),
            content_sha256=_hash_json_sha256(content),
            judgment_count=distinct_judgment_count,
            calibration_example_count=calibration_example_count,
            calibration_judgment_count=calibration_judgment_count,
        )


class PromptVariantMetadata(_MetadataModel):
    name: str
    system_sha256: str | None = None
    user_sha256: str

    @classmethod
    def from_content(
        cls, variant: Mapping[str, str | None]
    ) -> PromptVariantMetadata:
        return cls(
            name=variant["name"],
            system_sha256=_hash_string_sha256(variant.get("system_prompt")),
            user_sha256=_hash_string_sha256(variant.get("user_prompt_template")),
        )


class PromptMetadata(_MetadataModel):
    preset: str
    source: str
    parser_mode: str
    delegated: bool
    system_path: str | None = None
    user_path: str | None = None
    system_sha256: str | None = None
    user_sha256: str | None = None
    variants: list[PromptVariantMetadata] | None = None

    @classmethod
    def from_resolved(
        cls,
        prompt: ResolvedJudgePrompt,
        *,
        variants: Sequence[Mapping[str, str | None]] | None = None,
    ) -> PromptMetadata:
        """Fingerprint the exact prompt used by the judge."""
        variant_metadata = (
            [
                PromptVariantMetadata.from_content(variant)
                for variant in sorted(variants, key=lambda item: str(item["name"]))
            ]
            if variants
            else None
        )
        return cls(
            preset=prompt.preset_name,
            source=prompt.source,
            parser_mode=prompt.parser_mode,
            delegated=prompt.delegated,
            system_path=prompt.system_path,
            user_path=prompt.user_path,
            system_sha256=None if variant_metadata else prompt.system_sha256,
            user_sha256=None if variant_metadata else prompt.user_sha256,
            variants=variant_metadata,
        )


class CodeMetadata(_MetadataModel):
    git_commit: str | None = None
    git_dirty: bool | None = None


class EnvironmentMetadata(_MetadataModel):
    python_version: str
    platform: str
    dependencies: dict[str, str | None]


class ArtifactMetadata(_MetadataModel):
    kind: str
    path: str
    size_bytes: int
    sha256: str


class RunMetadata(_MetadataModel):
    schema_version: str
    identity: RunIdentity
    configuration: ConfigurationMetadata | None = None
    inputs: InputMetadata
    prompt: PromptMetadata | None = None
    code: CodeMetadata
    environment: EnvironmentMetadata
    execution: ExecutionMetadata
    artifacts: list[ArtifactMetadata]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_jsonable(value: Any) -> Any:
    """Convert arbitrary objects into JSON-safe values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        # JSON standard does not support NaN/Inf; encode as null.
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    # numpy / pandas scalars usually expose .item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass
    return str(value)


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(
        _to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _hash_string_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_json_sha256(value: Any) -> str | None:
    if value is None:
        return None
    return _hash_string_sha256(_stable_json_dumps(value))


def _hash_normalized_set_sha256(values: list[Any] | None) -> str | None:
    """Hash a collection independently of input order and duplicate values."""
    if values is None:
        return None

    normalized_by_key: dict[str, Any] = {}
    for value in values:
        normalized = _to_jsonable(value)
        normalized_by_key[_stable_json_dumps(normalized)] = normalized

    normalized_values = [normalized_by_key[key] for key in sorted(normalized_by_key)]
    return _hash_string_sha256(_stable_json_dumps(normalized_values))


def _hash_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_dist_name(requirement_spec: str) -> str | None:
    match = _REQUIREMENT_NAME_RE.match(requirement_spec or "")
    if not match:
        return None
    return match.group(1)


def _dependency_names_from_pyproject(repo_root: Path) -> list[str]:
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return []

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return []

    project = data.get("project", {})
    names: set[str] = set()

    for spec in project.get("dependencies", []) or []:
        dist = _extract_dist_name(spec)
        if dist:
            names.add(dist)

    optional = project.get("optional-dependencies", {}) or {}
    for specs in optional.values():
        for spec in specs or []:
            dist = _extract_dist_name(spec)
            if dist:
                names.add(dist)

    return sorted(names)


def _project_dependency_names(start_path: Path) -> list[str]:
    names: set[str] = set()

    # Prefer installed project metadata when available.
    try:
        dist = importlib_metadata.distribution("llm-judge-eval")
        for req in dist.requires or []:
            dep = _extract_dist_name(req)
            if dep:
                names.add(dep)
    except Exception:
        pass

    if names:
        return sorted(names)

    # Fallback: parse pyproject dependencies from repo root.
    repo_root = _run_git(["rev-parse", "--show-toplevel"], cwd=start_path)
    if repo_root is None:
        return []
    return _dependency_names_from_pyproject(Path(repo_root))


def _get_dependency_versions(
    dependencies: list[str] | None = None,
    start_path: Path | None = None,
) -> dict[str, str | None]:
    dep_names = dependencies or _project_dependency_names(start_path or Path.cwd())
    versions: dict[str, str | None] = {}
    for dist_name in dep_names:
        try:
            versions[dist_name] = importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            versions[dist_name] = None
        except Exception:
            versions[dist_name] = None
    return versions


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _get_git_hash(start_path: Path) -> str | None:
    repo_root = _run_git(["rev-parse", "--show-toplevel"], cwd=start_path)
    if repo_root is None:
        return None

    root = Path(repo_root)
    return _run_git(["rev-parse", "HEAD"], cwd=root)


def _get_git_dirty(start_path: Path) -> bool | None:
    repo_root = _run_git(["rev-parse", "--show-toplevel"], cwd=start_path)
    if repo_root is None:
        return None
    status = _run_git(
        ["status", "--porcelain", "--untracked-files=no"], cwd=Path(repo_root)
    )
    return bool(status) if status is not None else None


def _describe_artifacts(
    output_dir: Path,
    artifacts: Mapping[str, str | Path] | None,
) -> list[dict[str, Any]]:
    """Describe only files explicitly registered by the producing workflow."""
    described: list[dict[str, Any]] = []
    output_root = output_dir.resolve()
    for kind, raw_path in sorted((artifacts or {}).items()):
        path = Path(raw_path)
        candidates = [path.resolve()]
        if not path.is_absolute():
            candidates.append((output_dir / path).resolve())
        resolved_path = None
        rel = None
        for candidate in candidates:
            try:
                rel = candidate.relative_to(output_root)
                resolved_path = candidate
                break
            except ValueError:
                continue
        if resolved_path is None or rel is None:
            raise ValueError(
                f"Artifact '{kind}' must be inside output directory {output_dir}."
            )
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Artifact '{kind}' does not exist: {path}")
        described.append(
            {
                "kind": kind,
                "path": str(rel),
                "size_bytes": resolved_path.stat().st_size,
                "sha256": _hash_file_sha256(resolved_path),
            }
        )
    return described


def write_run_metadata(
    *,
    output_dir: str | Path,
    entrypoint: str,
    context: RunContext,
    inputs: InputMetadata | None = None,
    judge_prompt: ResolvedJudgePrompt | None = None,
    prompt_variants: Sequence[Mapping[str, str | None]] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    started_at_utc: datetime | None = None,
    metadata_filename: str = METADATA_FILENAME,
) -> Path:
    """Write run metadata JSON and return the output path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    finished = _utc_now()
    started = started_at_utc or finished
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    duration_sec = max(0.0, (finished - started).total_seconds())

    source_path = Path(__file__).resolve().parent
    metadata = RunMetadata(
        schema_version=METADATA_SCHEMA_VERSION,
        identity=context.identity,
        configuration=context.configuration,
        inputs=inputs or InputMetadata(),
        prompt=(
            PromptMetadata.from_resolved(judge_prompt, variants=prompt_variants)
            if judge_prompt is not None
            else None
        ),
        code=CodeMetadata(
            git_commit=_get_git_hash(start_path=source_path),
            git_dirty=_get_git_dirty(start_path=source_path),
        ),
        environment=EnvironmentMetadata(
            python_version=platform.python_version(),
            platform=platform.platform(),
            dependencies=_get_dependency_versions(start_path=source_path),
        ),
        execution=ExecutionMetadata(
            entrypoint=entrypoint,
            started_at_utc=started.isoformat(),
            finished_at_utc=finished.isoformat(),
            duration_seconds=duration_sec,
        ),
        artifacts=[
            ArtifactMetadata(**artifact)
            for artifact in _describe_artifacts(output_path, artifacts)
        ],
    )

    metadata_path = output_path / metadata_filename
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            _to_jsonable(metadata.model_dump(exclude_none=True)),
            f,
            indent=2,
            allow_nan=False,
        )
    return metadata_path
