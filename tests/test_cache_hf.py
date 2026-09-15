import shutil
from types import SimpleNamespace

import pandas as pd
from huggingface_hub.errors import HfHubHTTPError
from requests import Response

import judgearena.cache_hf as cache_hf
from judgearena.cache_sqlite import (
    COMPLETION_DB_NAME,
    CompletionCache,
    cache_folder,
    write_descriptor,
)

DESCRIPTOR = {"provider": "Dummy", "model": "model"}


def _create_cache(root, task, model, prompts):
    folder = cache_folder(root, "completions", task, model, DESCRIPTOR)
    write_descriptor(folder, DESCRIPTOR)
    rows = pd.DataFrame(
        [
            {
                "input_text": prompt,
                "completion": f"answer:{prompt}",
                "benchmark": task,
                "instruction_id": str(index),
                "model": model,
            }
            for index, prompt in enumerate(prompts)
        ]
    )
    with CompletionCache(folder / COMPLETION_DB_NAME) as cache:
        cache.save(rows, pushed_by="test")
    return folder


class FakeHub:
    def __init__(self, remote_root):
        self.remote_root = remote_root
        self.sha = "head-0"
        self.commits = []
        self.on_first_commit = None

    def repo_info(self, **_kwargs):
        return SimpleNamespace(sha=self.sha)

    def list_repo_files(self, **_kwargs):
        return [
            path.relative_to(self.remote_root).as_posix()
            for path in self.remote_root.rglob("*")
            if path.is_file()
        ]

    def create_commit(self, *, operations, parent_commit, **_kwargs):
        operations = list(operations)
        self.commits.append((parent_commit, operations))
        if self.on_first_commit is not None:
            callback, self.on_first_commit = self.on_first_commit, None
            callback()
            self.sha = "head-1"
            response = Response()
            response.status_code = 409
            raise HfHubHTTPError("conflict", response=response)
        for operation in operations:
            destination = self.remote_root / operation.path_in_repo
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(operation.path_or_fileobj, destination)
        self.sha = "head-2"
        return SimpleNamespace(oid=self.sha)


def _stub_download(monkeypatch, remote_root):
    def download(*, filename, local_dir, **_kwargs):
        destination = local_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_root / filename, destination)
        return str(destination)

    monkeypatch.setattr(cache_hf, "hf_hub_download", download)


def test_fetch_cache_applies_task_and_model_filters(tmp_path, monkeypatch):
    remote_root = tmp_path / "remote"
    wanted = _create_cache(remote_root, "arena-hard", "Dummy/model", ["wanted"])
    _create_cache(remote_root, "other-task", "Dummy/model", ["other"])
    api = FakeHub(remote_root)
    _stub_download(monkeypatch, remote_root)

    count = cache_hf.fetch_cache(
        tmp_path / "local",
        "org/cache",
        kind="completions",
        task="arena-hard",
        model_spec="Dummy/model",
        api=api,
    )

    local_db = tmp_path / "local" / wanted.relative_to(remote_root) / COMPLETION_DB_NAME
    assert count == 1
    assert CompletionCache(local_db).query()["completion"].tolist() == ["answer:wanted"]


def test_merge_cache_folder_uses_newer_row(tmp_path):
    local = _create_cache(tmp_path / "local", "task", "Dummy/model", ["same"])
    remote = _create_cache(tmp_path / "remote", "task", "Dummy/model", ["same"])
    with CompletionCache(remote / COMPLETION_DB_NAME) as cache:
        cache.save(
            pd.DataFrame(
                [
                    {
                        "input_text": "same",
                        "completion": "newer",
                        "benchmark": "task",
                        "instruction_id": "0",
                        "model": "Dummy/model",
                    }
                ]
            ),
            pushed_by="remote",
        )

    cache_hf.merge_cache_folder(local, remote, kind="completions")

    assert CompletionCache(local / COMPLETION_DB_NAME).query()[
        "completion"
    ].tolist() == ["newer"]


def test_push_remerges_after_parent_conflict_and_uploads_atomically(
    tmp_path, monkeypatch
):
    local_root = tmp_path / "local"
    remote_root = tmp_path / "remote"
    local = _create_cache(local_root, "task", "Dummy/model", ["local"])
    remote = _create_cache(remote_root, "task", "Dummy/model", ["remote"])
    api = FakeHub(remote_root)
    _stub_download(monkeypatch, remote_root)
    api.on_first_commit = lambda: _create_cache(
        remote_root, "task", "Dummy/model", ["concurrent"]
    )

    cache_hf.push_cache_folder(
        local_root,
        local,
        "org/cache",
        kind="completions",
        api=api,
    )

    assert [parent for parent, _ in api.commits] == ["head-0", "head-1"]
    assert all(len(operations) == 2 for _, operations in api.commits)
    outputs = (
        CompletionCache(remote / COMPLETION_DB_NAME).query()["completion"].tolist()
    )
    assert sorted(outputs) == ["answer:concurrent", "answer:local", "answer:remote"]
