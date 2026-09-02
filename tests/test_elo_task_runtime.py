from __future__ import annotations

import pandas as pd

import judgearena.datasets.arena_battles as arena_battles
from judgearena.tasks.registry import get_packaged_task


def _elo_task(task_id: str = "elo-comparia"):
    task = get_packaged_task(task_id)
    assert task is not None
    return task


def test_elo_dataset_adapter_uses_task_owned_arena_and_sources(monkeypatch, tmp_path):
    task = _elo_task()
    captured = {}

    def fake_load_arena_dataframe(arena, *, dataset_sources):
        captured["arena"] = arena
        captured["sources"] = dataset_sources
        return pd.DataFrame([{"question_id": "q1"}])

    monkeypatch.setattr(
        arena_battles, "load_arena_dataframe", fake_load_arena_dataframe
    )

    battles = arena_battles.load_task_battles(task, tmp_path)

    source = captured["sources"]["ministere-culture/comparia-votes"]
    assert captured["arena"] == "ComparIA"
    assert source.revision == "7a40bce496c1f2aa3be4001da85a49cb4743042b"
    assert battles["question_id"].tolist() == ["q1"]


def test_elo_dataset_download_uses_pinned_task_source(monkeypatch, tmp_path):
    task = _elo_task("elo-lmarena-100k")
    captured = {}
    monkeypatch.setattr(
        arena_battles,
        "snapshot_download",
        lambda **kwargs: captured.update(kwargs),
    )

    arena_battles.download_task_sources(task, tmp_path)

    assert captured["repo_id"] == "lmarena-ai/arena-human-preference-100k"
    assert captured["revision"] == "72e85b3ddc9c81bf7b659d6b03d4126dfd8fb34a"
    assert captured["allow_patterns"] == ("data/*.parquet",)
