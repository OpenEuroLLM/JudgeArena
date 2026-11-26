from __future__ import annotations

import pandas as pd
import pytest
import yaml

import judgearena.datasets._arena as arena_io
from judgearena.config import load_config
from judgearena.datasets.registry import resolve_dataset_adapter
from judgearena.tasks.registry import get_packaged_task


def _elo_task(task_id: str = "elo-comparia"):
    task = get_packaged_task(task_id)
    assert task is not None
    return task


def test_elo_dataset_adapter_uses_task_owned_arena_and_sources(monkeypatch, tmp_path):
    task = _elo_task()
    captured = {}

    def download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    pair = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    pd.DataFrame(
        [
            dict(
                id="q1",
                timestamp=0,
                conversation_a=pair,
                conversation_b=pair,
                model_a_name="a",
                model_b_name="b",
                chosen_model_name="a",
                both_equal=False,
            )
        ]
    ).to_parquet(tmp_path / "votes.parquet")
    monkeypatch.setattr(arena_io, "snapshot_download", download)
    monkeypatch.setattr(arena_io, "detect_language", lambda _: "en")
    battles = resolve_dataset_adapter(task, tmp_path).load_data().battles
    assert captured["repo_id"] == "ministere-culture/comparia-votes"
    assert captured["revision"] == "7a40bce496c1f2aa3be4001da85a49cb4743042b"
    assert battles["question_id"].tolist() == ["q1"]


def test_elo_dataset_download_uses_pinned_task_source(monkeypatch, tmp_path):
    task = _elo_task("elo-lmarena-100k")
    captured = {}
    monkeypatch.setattr(
        arena_io,
        "snapshot_download",
        lambda **kwargs: captured.update(kwargs),
    )

    resolve_dataset_adapter(task, tmp_path).download()

    assert captured["repo_id"] == "lmarena-ai/arena-human-preference-100k"
    assert captured["revision"] == "72e85b3ddc9c81bf7b659d6b03d4126dfd8fb34a"
    assert captured["allow_patterns"] == ("data/*.parquet",)


def test_combined_lmarena_reads_all_releases_in_replay_order(monkeypatch, tmp_path):
    turns = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    common = {"id": 1, "model_a": "a", "model_b": "b"}
    for release in ("55k", "100k", "140k"):
        folder = tmp_path / release
        (folder / "data").mkdir(parents=True)
        if release == "55k":
            row = dict(
                common,
                prompt="Q",
                response_a="A",
                response_b="B",
                winner_tie=1,
                winner_model_a=0,
                winner_model_b=0,
            )
            pd.DataFrame([row]).to_csv(folder / "train.csv", index=False)
        else:
            row = dict(
                common, conversation_a=turns, conversation_b=turns, winner="model_a"
            )
            if release == "100k":
                row["tstamp"] = 1
            else:
                row.update(timestamp=pd.Timestamp("2025-01-01"), winner="both_bad")
            pd.DataFrame([row]).to_parquet(folder / "data" / "train.parquet")
    monkeypatch.setattr(
        arena_io,
        "snapshot_download",
        lambda repo_id, **_: tmp_path / repo_id.rsplit("-", 1)[1],
    )
    monkeypatch.setattr(arena_io, "detect_language", lambda _: "en")
    data = resolve_dataset_adapter(_elo_task("elo-lmarena"), tmp_path).load_data()
    assert data.battles["benchmark"].tolist() == [
        "LMArena-100k",
        "LMArena-55k",
        "LMArena-140k",
    ]
    assert data.battles["reference_pref"].tolist() == [0.0, 0.5, 0.5]
    assert data.battles["battle_id"].tolist() == [
        f"LMArena:1:row-{i}" for i in range(3)
    ]

    current = resolve_dataset_adapter(
        _elo_task("elo-lmarena-140k"), tmp_path
    ).load_data()
    assert current.battles["battle_id"].tolist() == ["LMArena-140k:1"]

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "task": "elo-lmarena",
                "model": {"name": "Dummy/candidate"},
                "judge": {"model": "Dummy/judge"},
                "dataset": {"format": "LMArena-140k", "battle_columns": {}},
            }
        )
    )
    with pytest.raises(ValueError) as error:
        load_config(path)
    for key in ("format", "battle_columns"):
        assert key in str(error.value)
