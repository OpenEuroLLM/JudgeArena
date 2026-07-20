"""Hugging Face Hub synchronization for unified inference cache cells."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path, PurePosixPath

import pandas as pd
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from judgearena.log import get_logger
from judgearena.store_sqlite import (
    INFERENCE_COLUMNS,
    INFERENCE_DB_NAME,
    METADATA_COLUMNS,
    sanitize_path_component,
    write_store_metadata,
)

logger = get_logger(__name__)

DEFAULT_CACHE_REPO = "judge-arena/judge-arena-cache"
DEFAULT_INFERENCE_PREFIX = "inference"
_DEFAULT_MAX_RETRIES = 5
_INFERENCE_TABLE = "inference"
_METADATA_TABLE = "inference_metadata"


def _output_rank(output_text: str) -> str:
    return hashlib.sha256(str(output_text).encode("utf-8")).hexdigest()


def _remove_sidecars(db_path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        Path(f"{db_path}{suffix}").unlink(missing_ok=True)


def _copy_database(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    _remove_sidecars(destination)
    with (
        sqlite3.connect(source) as source_conn,
        sqlite3.connect(destination) as destination_conn,
    ):
        source_conn.backup(destination_conn)


def _read_table(db_path: Path, table: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(f'SELECT * FROM "{table}"', conn)


def _merge_inference_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=list(INFERENCE_COLUMNS))
    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        return merged
    merged = merged.copy()
    merged["_output_rank"] = merged["output_text"].map(_output_rank)
    merged = merged.sort_values(
        ["pushed_at", "run_id", "_output_rank"],
        kind="stable",
        na_position="first",
    )
    merged = merged.drop_duplicates(subset=["input_hash"], keep="last")
    return merged.drop(columns=["_output_rank"])


def _merge_metadata_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=list(METADATA_COLUMNS))
    merged = pd.concat(frames, ignore_index=True)
    if merged.empty:
        return merged
    merged = merged.sort_values(
        ["observed_at", "run_id"],
        kind="stable",
        na_position="first",
    )
    return merged.drop_duplicates(subset=["input_hash", "metadata_hash"], keep="last")


def _write_merged_db(
    inference: pd.DataFrame,
    metadata: pd.DataFrame,
    destination: Path,
    *,
    template: Path,
) -> None:
    _copy_database(template, destination)
    with sqlite3.connect(destination) as conn:
        conn.execute(f'DELETE FROM "{_INFERENCE_TABLE}"')
        conn.execute(f'DELETE FROM "{_METADATA_TABLE}"')
        if not inference.empty:
            inference = inference.astype(object).where(pd.notna(inference), None)
            conn.executemany(
                f'INSERT INTO "{_INFERENCE_TABLE}" '
                f"({', '.join(INFERENCE_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in INFERENCE_COLUMNS)})",
                inference.loc[:, list(INFERENCE_COLUMNS)].itertuples(
                    index=False,
                    name=None,
                ),
            )
        if not metadata.empty:
            metadata = metadata.astype(object).where(pd.notna(metadata), None)
            conn.executemany(
                f'INSERT INTO "{_METADATA_TABLE}" '
                f"({', '.join(METADATA_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in METADATA_COLUMNS)})",
                metadata.loc[:, list(METADATA_COLUMNS)].itertuples(
                    index=False,
                    name=None,
                ),
            )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _merge_dbs(sources: list[Path], destination: Path) -> None:
    """Merge inference rows and union metadata associations."""
    if not sources:
        raise ValueError("sources must not be empty")
    sources = [Path(source) for source in sources]
    inference_frames = [_read_table(source, _INFERENCE_TABLE) for source in sources]
    metadata_frames = [_read_table(source, _METADATA_TABLE) for source in sources]
    merged_inference = _merge_inference_frames(inference_frames)
    merged_metadata = _merge_metadata_frames(metadata_frames)
    _write_merged_db(
        merged_inference,
        merged_metadata,
        destination,
        template=sources[0],
    )


def _replace_db(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.unlink(missing_ok=True)
    _remove_sidecars(destination)
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    _remove_sidecars(destination)


def _head_oid(
    api: HfApi,
    repo_id: str,
    *,
    repo_type: str,
    revision: str,
) -> str | None:
    try:
        info = api.repo_info(
            repo_id,
            repo_type=repo_type,
            revision=revision,
        )
    except RepositoryNotFoundError:
        return None
    return getattr(info, "sha", None) or getattr(info, "oid", None)


def _download_remote_file(
    repo_id: str,
    path_in_repo: str,
    *,
    repo_type: str,
    revision: str,
    token: str | None,
) -> Path | None:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )
    except EntryNotFoundError:
        return None
    return Path(path)


def _cell_db_path(cell_dir: Path | str) -> Path:
    return Path(cell_dir) / INFERENCE_DB_NAME


def _cell_metadata_path(cell_dir: Path | str) -> Path:
    return Path(cell_dir) / "metadata.json"


def validate_path_filters(
    *,
    prefix: str | None = None,
    task: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    config_hash: str | None = None,
) -> str | None:
    """Validate hierarchical filters and return a normalized path prefix."""
    if prefix:
        normalized = prefix.strip("/")
        parts = PurePosixPath(normalized).parts
        if (
            not parts
            or parts[0] != DEFAULT_INFERENCE_PREFIX
            or len(parts) > 5
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError(
                f"--prefix must be a cache-cell directory under "
                f"{DEFAULT_INFERENCE_PREFIX!r}, got {normalized!r}."
            )
        return normalized
    if config_hash and not (task and provider and model):
        raise ValueError("--config_hash requires --task, --provider, and --model.")
    if model and not (task and provider):
        raise ValueError("--model requires --task and --provider.")
    if provider and not task:
        raise ValueError("--provider requires --task.")
    if not task:
        return None
    parts = [
        DEFAULT_INFERENCE_PREFIX,
        sanitize_path_component(task),
    ]
    if provider:
        parts.append(sanitize_path_component(provider))
    if model:
        parts.append(sanitize_path_component(model))
    if config_hash:
        parts.append(sanitize_path_component(config_hash))
    return "/".join(parts)


def rel_path_in_repo(local_path: Path | str, store_root: Path | str) -> str:
    """Return a cell path relative to the store root."""
    return (
        Path(local_path)
        .expanduser()
        .relative_to(Path(store_root).expanduser())
        .as_posix()
    )


def _local_path_for_remote_cell(store_root: Path, path_in_repo: str) -> Path:
    remote_path = PurePosixPath(path_in_repo)
    if (
        remote_path.is_absolute()
        or len(remote_path.parts) != 6
        or remote_path.parts[0] != DEFAULT_INFERENCE_PREFIX
        or remote_path.name != INFERENCE_DB_NAME
        or any(part in {"", ".", ".."} for part in remote_path.parts)
    ):
        raise ValueError(f"Invalid remote cache cell path: {path_in_repo!r}")
    local_path = store_root.joinpath(*remote_path.parts).resolve()
    local_path.relative_to(store_root.resolve())
    return local_path


def _path_matches_prefix(rel_path: str, normalized_prefix: str) -> bool:
    """Return True when *rel_path* equals or extends *normalized_prefix*."""
    return rel_path == normalized_prefix or rel_path.startswith(f"{normalized_prefix}/")


def iter_cell_dbs(
    store_root: Path | str,
    *,
    path_prefix: str | None = None,
) -> list[Path]:
    """Return local inference.db cells, optionally filtered by path prefix."""
    root = Path(store_root).expanduser()
    if not root.exists():
        return []
    normalized_prefix = path_prefix.strip("/") if path_prefix else None
    cells: list[Path] = []
    for candidate in sorted(root.rglob(INFERENCE_DB_NAME)):
        relative = rel_path_in_repo(candidate, root)
        try:
            expected = _local_path_for_remote_cell(root, relative)
        except ValueError:
            logger.warning("Ignoring noncanonical local cache cell: %s", candidate)
            continue
        if candidate.resolve() == expected:
            cells.append(candidate)
    if normalized_prefix is None:
        return cells
    return [
        cell
        for cell in cells
        if _path_matches_prefix(rel_path_in_repo(cell, root), normalized_prefix)
    ]


def discover_remote_cell_dbs(
    repo_id: str,
    *,
    path_prefix: str | None = None,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | None = None,
) -> list[str]:
    """List remote inference.db paths under an optional prefix."""
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(
            repo_id,
            repo_type=repo_type,
            revision=revision,
        )
    except RepositoryNotFoundError:
        return []
    normalized_prefix = path_prefix.strip("/") if path_prefix else None
    db_paths = [
        path
        for path in files
        if path.endswith(f"/{INFERENCE_DB_NAME}") or path == INFERENCE_DB_NAME
    ]
    if normalized_prefix is None:
        return sorted(db_paths)
    return sorted(
        path for path in db_paths if _path_matches_prefix(path, normalized_prefix)
    )


def _metadata_path_in_repo(db_path_in_repo: str) -> str:
    return str(Path(db_path_in_repo).parent / "metadata.json").replace("\\", "/")


def _materialize_remote_metadata(
    remote_metadata: Path,
    local_metadata_path: Path,
) -> None:
    """Validate or write the remote descriptor before any DB merge."""
    remote_config = json.loads(remote_metadata.read_text(encoding="utf-8"))
    write_store_metadata(local_metadata_path.parent, remote_config)


def fetch_cell(
    repo_id: str,
    path_in_repo: str,
    local_db_path: Path | str,
    *,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | None = None,
) -> bool:
    """Fetch a remote cell after validating sibling metadata."""
    local_db_path = Path(local_db_path)
    local_metadata_path = _cell_metadata_path(local_db_path.parent)
    metadata_in_repo = _metadata_path_in_repo(path_in_repo)

    remote_db = _download_remote_file(
        repo_id,
        path_in_repo,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    if remote_db is None:
        return False

    remote_metadata = _download_remote_file(
        repo_id,
        metadata_in_repo,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    if remote_metadata is None:
        raise ValueError(
            f"Remote cell {path_in_repo} is missing required {metadata_in_repo}."
        )

    local_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _materialize_remote_metadata(remote_metadata, local_metadata_path)

    if local_db_path.exists():
        with tempfile.TemporaryDirectory() as temporary:
            merged = Path(temporary) / "merged.db"
            _merge_dbs([local_db_path, remote_db], merged)
            _replace_db(local_db_path, merged)
    else:
        _copy_database(remote_db, local_db_path)

    logger.info("Fetched %s from %s", path_in_repo, repo_id)
    return True


def fetch_cell_metadata(
    repo_id: str,
    metadata_path_in_repo: str,
    local_metadata_path: Path | str,
    *,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | None = None,
) -> bool:
    """Download remote metadata.json when present."""
    local_metadata_path = Path(local_metadata_path)
    remote = _download_remote_file(
        repo_id,
        metadata_path_in_repo,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    if remote is None:
        return False
    _materialize_remote_metadata(remote, local_metadata_path)
    logger.info("Fetched %s from %s", metadata_path_in_repo, repo_id)
    return True


def fetch_cells(
    repo_id: str,
    store_root: Path | str,
    db_paths: list[Path | str],
    *,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | None = None,
    strict: bool = False,
) -> None:
    """Fetch cells and sibling metadata, warning instead of failing by default."""
    for db_path_value in db_paths:
        db_path = Path(db_path_value)
        path_in_repo = rel_path_in_repo(db_path, store_root)
        try:
            fetch_cell(
                repo_id,
                path_in_repo,
                db_path,
                repo_type=repo_type,
                revision=revision,
                token=token,
            )
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            logger.warning("Cache fetch skipped for %s: %s", path_in_repo, exc)


def fetch_remote_cells(
    repo_id: str,
    store_root: Path | str,
    *,
    path_prefix: str | None = None,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | None = None,
    strict: bool = False,
) -> list[Path]:
    """Discover and fetch remote cells into a possibly empty local store."""
    store_root = Path(store_root).expanduser()
    store_root.mkdir(parents=True, exist_ok=True)
    remote_paths = discover_remote_cell_dbs(
        repo_id,
        path_prefix=path_prefix,
        repo_type=repo_type,
        revision=revision,
        token=token,
    )
    local_paths: list[Path] = []
    for path_in_repo in remote_paths:
        try:
            local_paths.append(_local_path_for_remote_cell(store_root, path_in_repo))
        except ValueError as exc:
            if strict:
                raise
            logger.warning("Remote cache cell skipped: %s", exc)
    fetch_cells(
        repo_id,
        store_root,
        local_paths,
        repo_type=repo_type,
        revision=revision,
        token=token,
        strict=strict,
    )
    return local_paths


def push_cell(
    repo_id: str,
    path_in_repo: str,
    local_db_path: Path | str,
    *,
    pushed_by: str,
    local_metadata_path: Path | str | None = None,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | None = None,
    create_pr: bool = False,
    ensure_repo: bool = False,
    private: bool = True,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> str:
    """Merge and upload a cell DB plus metadata in one optimistic commit."""
    local_db_path = Path(local_db_path)
    if not local_db_path.exists():
        raise FileNotFoundError(local_db_path)

    metadata_path = (
        Path(local_metadata_path)
        if local_metadata_path is not None
        else local_db_path.parent / "metadata.json"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata.json for cache cell at {local_db_path}"
        )
    metadata_in_repo = _metadata_path_in_repo(path_in_repo)

    api = HfApi(token=token)
    if ensure_repo:
        api.create_repo(
            repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )

    last_error: HfHubHTTPError | None = None
    for attempt in range(max_retries):
        parent = _head_oid(
            api,
            repo_id,
            repo_type=repo_type,
            revision=revision,
        )
        remote = _download_remote_file(
            repo_id,
            path_in_repo,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )
        sources = [local_db_path] if remote is None else [remote, local_db_path]
        with tempfile.TemporaryDirectory() as temporary:
            merged = Path(temporary) / "merged.db"
            _merge_dbs(sources, merged)
            operations = [
                CommitOperationAdd(
                    path_in_repo=path_in_repo,
                    path_or_fileobj=str(merged),
                ),
                CommitOperationAdd(
                    path_in_repo=metadata_in_repo,
                    path_or_fileobj=str(metadata_path),
                ),
            ]
            try:
                info = api.create_commit(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    operations=operations,
                    parent_commit=parent,
                    create_pr=create_pr,
                    commit_message=(
                        f"cache: {Path(path_in_repo).parent.as_posix()} (by {pushed_by})"
                    ),
                )
            except HfHubHTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status == 412 and attempt < max_retries - 1:
                    last_error = exc
                    logger.info(
                        "Push of %s hit a concurrent commit (412); retrying (%d/%d).",
                        path_in_repo,
                        attempt + 1,
                        max_retries,
                    )
                    continue
                raise
            _replace_db(local_db_path, merged)

        result = (
            getattr(info, "pr_url", None)
            or getattr(info, "oid", None)
            or getattr(info, "sha", None)
            or ""
        )
        logger.info("Pushed %s to %s (%s)", path_in_repo, repo_id, result)
        return result

    raise RuntimeError(
        f"push_cell exhausted {max_retries} retries for {path_in_repo}"
    ) from last_error


def push_cells(
    repo_id: str,
    store_root: Path | str,
    db_paths: list[Path | str],
    *,
    pushed_by: str,
    repo_type: str = "dataset",
    revision: str = "main",
    token: str | None = None,
    create_pr: bool = False,
    ensure_repo: bool = False,
    private: bool = True,
    strict: bool = False,
) -> None:
    """Push existing local cells and their descriptor metadata."""
    for db_path_value in db_paths:
        db_path = Path(db_path_value)
        if not db_path.exists():
            message = f"Cache cell does not exist: {db_path}"
            if strict:
                raise FileNotFoundError(message)
            logger.warning(message)
            continue
        path_in_repo = rel_path_in_repo(db_path, store_root)
        try:
            push_cell(
                repo_id,
                path_in_repo,
                db_path,
                pushed_by=pushed_by,
                repo_type=repo_type,
                revision=revision,
                token=token,
                create_pr=create_pr,
                ensure_repo=ensure_repo,
                private=private,
            )
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise
            logger.warning("Cache push skipped for %s: %s", path_in_repo, exc)
