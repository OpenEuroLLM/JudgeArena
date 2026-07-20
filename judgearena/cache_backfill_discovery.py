"""Discover and classify saved run folders and cache artifacts for backfill."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml

from judgearena.constants import ELO_TASK_PREFIX, META_EVAL_TASK
from judgearena.repro import METADATA_FILENAME

LEGACY_CELL_DB_NAMES = frozenset({"judgements.db", "completions.db"})
PASS_LEVEL_CACHE_SUFFIXES = (".csv.zip", ".parquet", ".csv")
GENERATION_ONLY_MARKERS = frozenset(
    {
        "completions.parquet",
        "completions.csv",
        "completions.csv.zip",
        "model_outputs.parquet",
    }
)
GAE_REQUIRED_COLUMNS = frozenset(
    {"instruction", "completion_A", "completion_B", "judge_input"}
)
GAE_OUTPUT_COLUMNS = frozenset({"judge_completion", "judge_output"})


class ArtifactKind(StrEnum):
    GAE_RUN = "gae_run"
    MT_BENCH_RUN = "mt_bench_run"
    META_EVAL_RUN = "meta_eval_run"
    ELO_RUN = "elo_run"
    LEGACY_CACHE_CELL = "legacy_cache_cell"
    META_EVAL_IDENTITY_DB = "meta_eval_identity_db"
    PASS_LEVEL_CACHE = "pass_level_cache"
    GENERATION_ARTIFACT = "generation_artifact"
    UNKNOWN = "unknown"


SKIP_REASON_BY_KIND: dict[ArtifactKind, str] = {
    ArtifactKind.ELO_RUN: "elo_run_missing_inference_outputs",
    ArtifactKind.LEGACY_CACHE_CELL: "legacy_cache_cell_unmigratable",
    ArtifactKind.META_EVAL_IDENTITY_DB: "meta_eval_identity_db",
    ArtifactKind.PASS_LEVEL_CACHE: "pass_level_cache_untrusted",
    ArtifactKind.GENERATION_ARTIFACT: "generation_provenance_unknown",
    ArtifactKind.UNKNOWN: "unknown_judge_run",
}


@dataclass
class ClassifiedSource:
    path: Path
    kind: ArtifactKind
    run_dir: Path | None = None


@dataclass
class DiscoveryReport:
    migratable_runs: list[ClassifiedSource] = field(default_factory=list)
    skipped: list[ClassifiedSource] = field(default_factory=list)


def _glob_has_matches(resolved: Path, pattern: str) -> bool:
    return next(resolved.glob(pattern), None) is not None


def _is_elo_task_name(value: str | None) -> bool:
    return bool(value and value.startswith(ELO_TASK_PREFIX))


def _looks_like_meta_eval_dir(path: Path) -> bool:
    return (
        path.name.startswith(f"{META_EVAL_TASK}-")
        or (path / "annotations.parquet").exists()
    )


def _csv_columns(csv_path: Path) -> set[str]:
    header = csv_path.read_text(encoding="utf-8").splitlines()[:1]
    if not header:
        return set()
    return {part.strip() for part in header[0].split(",")}


def _looks_like_mt_annotations(path: Path) -> bool:
    for csv_path in path.glob("*-annotations.csv"):
        columns = _csv_columns(csv_path)
        if not columns:
            continue
        mt_markers = {"question_id", "turn", "category"}
        if mt_markers.issubset(columns):
            return True
        if "g1_user_prompt" in columns or "user_prompt" in columns:
            return True
    return False


def _looks_like_gae_annotations(path: Path) -> bool:
    for csv_path in path.glob("*-annotations.csv"):
        columns = _csv_columns(csv_path)
        if not columns:
            continue
        if not GAE_REQUIRED_COLUMNS.issubset(columns):
            continue
        if not GAE_OUTPUT_COLUMNS.intersection(columns):
            continue
        return True
    return False


def _task_from_run_dir(run_dir: Path) -> str | None:
    metadata_path = run_dir / METADATA_FILENAME
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        run = payload.get("run")
        if isinstance(run, dict):
            task = run.get("task")
            if isinstance(task, str):
                return task
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        task = payload.get("task")
        if isinstance(task, str):
            return task
    for args_path in run_dir.glob("args-*.json"):
        payload = json.loads(args_path.read_text(encoding="utf-8"))
        task = payload.get("task")
        if isinstance(task, str):
            return task
    args_path = run_dir / "args.json"
    if args_path.exists():
        payload = json.loads(args_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            task = payload.get("task")
            if isinstance(task, str):
                return task
    return None


def _classify_path(path: Path) -> ClassifiedSource:
    resolved = path.resolve()

    if resolved.is_file():
        suffix = "".join(resolved.suffixes) or resolved.suffix
        name = resolved.name
        if name in GENERATION_ONLY_MARKERS:
            return ClassifiedSource(resolved, ArtifactKind.GENERATION_ARTIFACT)
        if (
            name.endswith(PASS_LEVEL_CACHE_SUFFIXES)
            or suffix in PASS_LEVEL_CACHE_SUFFIXES
        ):
            return ClassifiedSource(resolved, ArtifactKind.PASS_LEVEL_CACHE)
        if resolved.suffix == ".db":
            parts = {part.lower() for part in resolved.parts}
            if "cache" in parts and "db" in parts:
                return ClassifiedSource(resolved, ArtifactKind.META_EVAL_IDENTITY_DB)
            if name in LEGACY_CELL_DB_NAMES:
                return ClassifiedSource(resolved, ArtifactKind.LEGACY_CACHE_CELL)
        return ClassifiedSource(resolved, ArtifactKind.UNKNOWN)

    if not resolved.is_dir():
        return ClassifiedSource(resolved, ArtifactKind.UNKNOWN)

    if resolved.name in LEGACY_CELL_DB_NAMES or any(
        (resolved / db_name).exists() for db_name in LEGACY_CELL_DB_NAMES
    ):
        return ClassifiedSource(resolved, ArtifactKind.LEGACY_CACHE_CELL)

    parts = {part.lower() for part in resolved.parts}
    if (
        "cache" in parts
        and "db" in parts
        and any(child.suffix == ".db" for child in resolved.glob("*.db"))
    ):
        return ClassifiedSource(resolved, ArtifactKind.META_EVAL_IDENTITY_DB)

    task = _task_from_run_dir(resolved)
    if _is_elo_task_name(task) or resolved.name.startswith(ELO_TASK_PREFIX):
        return ClassifiedSource(resolved, ArtifactKind.ELO_RUN, run_dir=resolved)

    if (resolved / "annotations.parquet").exists() and (
        (resolved / "args.json").exists() or (resolved / METADATA_FILENAME).exists()
    ):
        return ClassifiedSource(
            resolved,
            ArtifactKind.META_EVAL_RUN,
            run_dir=resolved,
        )

    if list(resolved.glob("*-annotations.csv")):
        if task == "mt-bench" or _looks_like_mt_annotations(resolved):
            return ClassifiedSource(
                resolved,
                ArtifactKind.MT_BENCH_RUN,
                run_dir=resolved,
            )
        if _is_elo_task_name(task):
            return ClassifiedSource(resolved, ArtifactKind.ELO_RUN, run_dir=resolved)
        if _looks_like_gae_annotations(resolved):
            return ClassifiedSource(resolved, ArtifactKind.GAE_RUN, run_dir=resolved)
        return ClassifiedSource(resolved, ArtifactKind.UNKNOWN)

    if _looks_like_meta_eval_dir(resolved) and (resolved / "args.json").exists():
        return ClassifiedSource(
            resolved,
            ArtifactKind.META_EVAL_RUN,
            run_dir=resolved,
        )

    if any(resolved.joinpath(name).exists() for name in GENERATION_ONLY_MARKERS):
        return ClassifiedSource(resolved, ArtifactKind.GENERATION_ARTIFACT)

    if any(
        _glob_has_matches(resolved, f"*{suffix}")
        for suffix in PASS_LEVEL_CACHE_SUFFIXES
    ):
        if not list(resolved.glob("*-annotations.csv")):
            return ClassifiedSource(resolved, ArtifactKind.PASS_LEVEL_CACHE)

    return ClassifiedSource(resolved, ArtifactKind.UNKNOWN)


def _discover_run_dirs(source: Path) -> list[Path]:
    classified = _classify_path(source)
    if classified.run_dir is not None:
        return [classified.run_dir]

    if not source.is_dir():
        return []

    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for annotation_csv in source.rglob("*-annotations.csv"):
        run_dir = annotation_csv.parent.resolve()
        if run_dir not in seen:
            seen.add(run_dir)
            run_dirs.append(run_dir)
    for annotation_parquet in source.rglob("annotations.parquet"):
        run_dir = annotation_parquet.parent.resolve()
        if run_dir not in seen:
            seen.add(run_dir)
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def _discover_nested_skipped_artifacts(source: Path) -> list[ClassifiedSource]:
    skipped: list[ClassifiedSource] = []
    seen: set[Path] = set()
    for db_name in LEGACY_CELL_DB_NAMES:
        for db_path in source.rglob(db_name):
            parent = db_path.parent.resolve()
            if parent in seen:
                continue
            seen.add(parent)
            classified = _classify_path(parent)
            if classified.kind in SKIP_REASON_BY_KIND:
                skipped.append(classified)
    return skipped


def _collect_skipped_artifacts(
    source: Path, seen_skipped: set[Path]
) -> list[ClassifiedSource]:
    skipped: list[ClassifiedSource] = []
    for child in source.rglob("*"):
        if child in seen_skipped:
            continue
        classified = _classify_path(child)
        if classified.kind not in SKIP_REASON_BY_KIND:
            continue
        seen_skipped.add(child)
        skipped.append(classified)
    return skipped


def discover_sources(sources: list[Path]) -> DiscoveryReport:
    """Discover migratable judge run folders and classify skipped artifacts."""
    report = DiscoveryReport()
    seen_runs: set[Path] = set()
    seen_skipped: set[Path] = set()

    for source in sources:
        source = source.resolve()
        direct = _classify_path(source)

        if direct.run_dir is not None:
            if direct.run_dir not in seen_runs:
                seen_runs.add(direct.run_dir)
                report.migratable_runs.append(direct)
            continue

        if source.is_file():
            if source not in seen_skipped:
                seen_skipped.add(source)
                report.skipped.append(direct)
            continue

        run_dirs = _discover_run_dirs(source)
        for nested in _discover_nested_skipped_artifacts(source):
            if nested.path not in seen_skipped:
                seen_skipped.add(nested.path)
                report.skipped.append(nested)

        if not run_dirs:
            for classified_child in _collect_skipped_artifacts(source, seen_skipped):
                report.skipped.append(classified_child)
            if (
                direct.kind in SKIP_REASON_BY_KIND
                and source not in seen_skipped
                and (source.is_file() or direct.kind != ArtifactKind.UNKNOWN)
            ):
                seen_skipped.add(source)
                report.skipped.append(direct)
            continue

        for run_dir in run_dirs:
            if run_dir in seen_runs:
                continue
            classified = _classify_path(run_dir)
            if classified.kind in SKIP_REASON_BY_KIND:
                if run_dir not in seen_skipped:
                    seen_skipped.add(run_dir)
                    report.skipped.append(classified)
                continue
            if classified.kind in {
                ArtifactKind.GAE_RUN,
                ArtifactKind.MT_BENCH_RUN,
                ArtifactKind.META_EVAL_RUN,
            }:
                seen_runs.add(run_dir)
                report.migratable_runs.append(classified)
            elif run_dir not in seen_skipped:
                seen_skipped.add(run_dir)
                report.skipped.append(classified)

    return report
