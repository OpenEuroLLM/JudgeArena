"""Filtered Hugging Face synchronization for inference-cache folders."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from judgearena.cache.sqlite import (
    COMPLETION_DB_NAME,
    DESCRIPTOR_FILENAME,
    JUDGEMENT_DB_NAME,
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


def _matches_cache_folder(
    parts: tuple[str, ...],
    *,
    kind: CacheKind,
    task: str | None,
    model_spec: str | None,
) -> bool:
    if len(parts) != 5 or parts[0] != kind:
        return False
    if task is not None and unquote(parts[1]) != task:
        return False
    if model_spec is not None:
        provider, model = model_spec.split("/", 1)
        if (unquote(parts[2]), unquote(parts[3])) != (provider, model):
            return False
    return True


def list_remote_cache_folders(
    api: HfApi,
    hf_repo: str,
    *,
    kind: CacheKind,
    task: str | None = None,
    model_spec: str | None = None,
    revision: str | None = None,
) -> list[str]:
    """List complete remote cache folders matching the supplied filters."""
    files = set(
        api.list_repo_files(
            repo_id=hf_repo,
            repo_type="dataset",
            revision=revision,
        )
    )
    db_name = _DB_NAMES[kind]
    return sorted(
        {
            str(PurePosixPath(path).parent)
            for path in files
            if PurePosixPath(path).name == DESCRIPTOR_FILENAME
            and _matches_cache_folder(
                PurePosixPath(path).parent.parts,
                kind=kind,
                task=task,
                model_spec=model_spec,
            )
            and str(PurePosixPath(path).parent / db_name) in files
        }
    )


def list_local_cache_folders(
    store_root: Path,
    *,
    kind: CacheKind,
    task: str | None = None,
    model_spec: str | None = None,
) -> list[Path]:
    """List complete local cache folders matching the supplied filters."""
    base = store_root / kind
    if not base.exists():
        return []
    db_name = _DB_NAMES[kind]
    return sorted(
        metadata.parent
        for metadata in base.rglob(DESCRIPTOR_FILENAME)
        if _matches_cache_folder(
            metadata.parent.relative_to(store_root).parts,
            kind=kind,
            task=task,
            model_spec=model_spec,
        )
        and (metadata.parent / db_name).exists()
    )


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
    kind: CacheKind,
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
    for repo_folder in folders:
        with tempfile.TemporaryDirectory() as temporary_dir:
            remote_folder = _download_cache_folder(
                hf_repo,
                repo_folder,
                kind=kind,
                revision=revision,
                local_dir=Path(temporary_dir),
            )
            merge_cache_folder(
                store_root / repo_folder,
                remote_folder,
                kind=kind,
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
    kind: CacheKind,
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
            folder,
            hf_repo,
            kind=kind,
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

    kinds = (args.kind,) if args.kind is not None else _CACHE_KINDS
    count = 0
    for kind in kinds:
        kwargs = {
            "kind": kind,
            "task": args.task,
            "model_spec": args.model_spec,
        }
        if args.action == "fetch":
            count += fetch_cache(args.store_root, args.hf_repo, **kwargs)
        else:
            count += len(
                push_cache(
                    args.store_root,
                    args.hf_repo,
                    **kwargs,
                )
            )
    print(f"{args.action.capitalize()}ed {count} cache folders.")
