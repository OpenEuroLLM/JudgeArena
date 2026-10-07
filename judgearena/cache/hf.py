"""Filtered Hugging Face synchronization for inference-cache folders."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from judgearena.cache.sqlite import (
    COMPLETION_DB_NAME,
    DESCRIPTOR_FILENAME,
    JUDGEMENT_DB_NAME,
    CacheFolder,
    CacheKind,
    CompletionCache,
    JudgementCache,
    read_descriptor,
    write_descriptor,
)

DEFAULT_HF_CACHE_REPO = "judge-arena/judge-arena-cache"
_DB_NAMES = {
    "completions": COMPLETION_DB_NAME,
    "judgements": JUDGEMENT_DB_NAME,
}
_CACHE_KINDS: tuple[CacheKind, ...] = ("completions", "judgements")
_STORE_TYPES = {
    "completions": CompletionCache,
    "judgements": JudgementCache,
}


def _find_cache_folders(
    files: set[str],
    *,
    kind: CacheKind | None,
    task: str | None,
    model_spec: str | None,
) -> list[CacheFolder]:
    """Find complete cache folders in relative file paths matching the filters."""
    folders = (
        CacheFolder.parse(PurePosixPath(file).parent)
        for file in files
        if PurePosixPath(file).name == DESCRIPTOR_FILENAME
    )
    return sorted(
        folder
        for folder in folders
        if folder is not None
        and folder.kind in _DB_NAMES
        and kind in (None, folder.kind)
        and task in (None, folder.task)
        and model_spec in (None, folder.model_spec)
        and f"{folder.path}/{_DB_NAMES[folder.kind]}" in files
    )


def list_remote_cache_folders(
    api: HfApi,
    hf_repo: str,
    *,
    kind: CacheKind | None = None,
    task: str | None = None,
    model_spec: str | None = None,
    revision: str | None = None,
) -> list[CacheFolder]:
    files = api.list_repo_files(repo_id=hf_repo, repo_type="dataset", revision=revision)
    return _find_cache_folders(set(files), kind=kind, task=task, model_spec=model_spec)


def list_local_cache_folders(
    store_root: Path,
    *,
    kind: CacheKind | None = None,
    task: str | None = None,
    model_spec: str | None = None,
) -> list[CacheFolder]:
    files = {
        path.relative_to(store_root).as_posix()
        for path in store_root.rglob("*")
        if path.is_file()
    }
    return _find_cache_folders(files, kind=kind, task=task, model_spec=model_spec)


def _download_cache_folder(
    hf_repo: str,
    repo_folder: str,
    *,
    kind: CacheKind,
    revision: str,
    local_dir: Path,
) -> Path:
    for filename in (DESCRIPTOR_FILENAME, _DB_NAMES[kind]):
        hf_hub_download(
            repo_id=hf_repo,
            repo_type="dataset",
            filename=f"{repo_folder}/{filename}",
            revision=revision,
            local_dir=local_dir,
        )
    return local_dir / repo_folder


def merge_cache_folder(
    local_folder: Path,
    remote_folder: Path,
    *,
    kind: CacheKind,
) -> int:
    """Merge a downloaded folder into a local cache using row provenance."""
    remote_descriptor = read_descriptor(remote_folder)
    local_metadata = local_folder / DESCRIPTOR_FILENAME
    if local_metadata.exists():
        if read_descriptor(local_folder) != remote_descriptor:
            raise ValueError(f"Descriptor mismatch for cache folder {local_folder}.")
    else:
        write_descriptor(local_folder, remote_descriptor)

    db_name = _DB_NAMES[kind]
    store = _STORE_TYPES[kind](local_folder / db_name)
    return store.merge_from(remote_folder / db_name)


def fetch_cache(
    store_root: Path,
    hf_repo: str,
    *,
    kind: CacheKind | None = None,
    task: str | None = None,
    model_spec: str | None = None,
    api: HfApi | None = None,
) -> int:
    """Fetch and merge matching cache folders from a dataset repository."""
    api = api or HfApi()
    revision = api.repo_info(repo_id=hf_repo, repo_type="dataset").sha
    folders = list_remote_cache_folders(
        api,
        hf_repo,
        kind=kind,
        task=task,
        model_spec=model_spec,
        revision=revision,
    )
    for folder in folders:
        with tempfile.TemporaryDirectory() as temporary_dir:
            remote_folder = _download_cache_folder(
                hf_repo,
                folder.path.as_posix(),
                kind=folder.kind,
                revision=revision,
                local_dir=Path(temporary_dir),
            )
            merge_cache_folder(
                store_root / folder.path,
                remote_folder,
                kind=folder.kind,
            )
    return len(folders)


def _is_commit_conflict(error: HfHubHTTPError) -> bool:
    return error.response is not None and error.response.status_code in {409, 412}


def push_cache_folder(
    store_root: Path,
    local_folder: Path,
    hf_repo: str,
    *,
    kind: CacheKind,
    api: HfApi | None = None,
    max_retries: int = 3,
) -> Any:
    """Pull, merge, and atomically push one folder with optimistic retry."""
    api = api or HfApi()
    repo_folder = local_folder.relative_to(store_root).as_posix()
    db_name = _DB_NAMES[kind]
    read_descriptor(local_folder)

    for attempt in range(max_retries):
        parent_commit = api.repo_info(repo_id=hf_repo, repo_type="dataset").sha
        files = set(
            api.list_repo_files(
                repo_id=hf_repo,
                repo_type="dataset",
                revision=parent_commit,
            )
        )
        if f"{repo_folder}/{DESCRIPTOR_FILENAME}" in files:
            with tempfile.TemporaryDirectory() as temporary_dir:
                remote_folder = _download_cache_folder(
                    hf_repo,
                    repo_folder,
                    kind=kind,
                    revision=parent_commit,
                    local_dir=Path(temporary_dir),
                )
                merge_cache_folder(local_folder, remote_folder, kind=kind)

        operations = [
            CommitOperationAdd(
                path_in_repo=f"{repo_folder}/{DESCRIPTOR_FILENAME}",
                path_or_fileobj=local_folder / DESCRIPTOR_FILENAME,
            ),
            CommitOperationAdd(
                path_in_repo=f"{repo_folder}/{db_name}",
                path_or_fileobj=local_folder / db_name,
            ),
        ]
        try:
            return api.create_commit(
                repo_id=hf_repo,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Sync JudgeArena cache {repo_folder}",
                parent_commit=parent_commit,
            )
        except HfHubHTTPError as error:
            if not _is_commit_conflict(error) or attempt == max_retries - 1:
                raise


def push_cache(
    store_root: Path,
    hf_repo: str,
    *,
    kind: CacheKind | None = None,
    task: str | None = None,
    model_spec: str | None = None,
    api: HfApi | None = None,
    max_retries: int = 3,
) -> list[Any]:
    """Push all matching local cache folders to a dataset repository."""
    api = api or HfApi()
    return [
        push_cache_folder(
            store_root,
            store_root / folder.path,
            hf_repo,
            kind=folder.kind,
            api=api,
            max_retries=max_retries,
        )
        for folder in list_local_cache_folders(
            store_root,
            kind=kind,
            task=task,
            model_spec=model_spec,
        )
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync JudgeArena inference caches.")
    parser.add_argument("--action", choices=["fetch", "push"], required=True)
    parser.add_argument("--store_root", type=Path, required=True)
    parser.add_argument("--hf_repo", default=DEFAULT_HF_CACHE_REPO)
    parser.add_argument("--kind", choices=_CACHE_KINDS)
    parser.add_argument("--task")
    parser.add_argument(
        "--model",
        dest="model_spec",
        help="Optional full Provider/model filter.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Synchronize every cache folder; cannot be combined with filters.",
    )
    return parser


def cli(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    filters = (args.kind, args.task, args.model_spec)
    if args.all and any(filters):
        parser.error("--all cannot be combined with --kind, --task, or --model.")
    if not args.all and not any(filters):
        parser.error("Specify a filter or use --all.")

    kwargs = {
        "kind": args.kind,
        "task": args.task,
        "model_spec": args.model_spec,
    }
    if args.action == "fetch":
        count = fetch_cache(args.store_root, args.hf_repo, **kwargs)
    else:
        count = len(push_cache(args.store_root, args.hf_repo, **kwargs))
    print(f"{args.action.capitalize()}ed {count} cache folders.")
