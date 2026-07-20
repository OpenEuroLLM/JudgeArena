import uuid
from pathlib import Path

import pytest
from huggingface_hub.utils import (
    EntryNotFoundError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from judgearena import store_sync


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict = {}
        self.text = ""
        self.reason = "Precondition Failed"
        self.request = None


class _Info:
    def __init__(self, oid=None, sha=None, pr_url=None) -> None:
        self.oid = oid
        self.sha = sha
        self.pr_url = pr_url


class _FakeRepo:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.head: str | None = None
        self.pending_conflict = False
        self.inject = None
        self.commit_calls = 0


class _FakeHfApi:
    def __init__(self, repo: _FakeRepo) -> None:
        self.repo = repo

    def create_repo(self, *args, **kwargs):
        return None

    def repo_info(self, repo_id, repo_type="dataset", revision="main"):
        if self.repo.head is None:
            raise RepositoryNotFoundError(
                f"Repository {repo_id} not found",
                response=_Response(404),
            )
        return _Info(sha=self.repo.head)

    def list_repo_files(self, repo_id, repo_type="dataset", revision="main"):
        if self.repo.head is None:
            raise RepositoryNotFoundError(
                f"Repository {repo_id} not found",
                response=_Response(404),
            )
        return sorted(self.repo.files)

    def create_commit(
        self,
        *,
        repo_id,
        repo_type="dataset",
        revision="main",
        operations,
        parent_commit=None,
        create_pr=False,
        commit_message=None,
    ):
        self.repo.commit_calls += 1
        if self.repo.pending_conflict:
            self.repo.pending_conflict = False
            if self.repo.inject is not None:
                self.repo.inject()
            raise HfHubHTTPError(
                "412 Precondition Failed",
                response=_Response(412),
            )
        if (
            not create_pr
            and parent_commit is not None
            and parent_commit != self.repo.head
        ):
            raise HfHubHTTPError(
                "412 Precondition Failed",
                response=_Response(412),
            )
        if create_pr:
            return _Info(pr_url="https://hf.co/pr/1")
        for operation in operations:
            self.repo.files[operation.path_in_repo] = Path(
                operation.path_or_fileobj
            ).read_bytes()
        self.repo.head = uuid.uuid4().hex
        return _Info(oid=self.repo.head, sha=self.repo.head)


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    repo = _FakeRepo()

    def fake_download(
        repo_id,
        filename,
        repo_type="dataset",
        revision="main",
        token=None,
    ):
        if filename not in repo.files:
            raise EntryNotFoundError(f"{filename} not found")
        destination = tmp_path / "downloads" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(repo.files[filename])
        return str(destination)

    monkeypatch.setattr(
        store_sync,
        "HfApi",
        lambda token=None: _FakeHfApi(repo),
    )
    monkeypatch.setattr(store_sync, "hf_hub_download", fake_download)
    return repo
