import json

import pytest

import judgearena.datasets.mt_bench_101 as mt_bench_101
from judgearena.datasets.mt_bench_101 import expand_mt_bench_101_records
from judgearena.tasks.registry import get_packaged_task


def test_expand_mt_bench_101_turn_rules():
    records = [
        {
            "task": "CM",
            "id": 1,
            "history": [
                {"user": "u1", "bot": "b1"},
                {"user": "u2", "bot": "b2"},
            ],
        },
        {
            "task": "PI",
            "id": 2,
            "history": [
                {"user": "x1", "bot": "y1"},
                {"user": "x2", "bot": "y2"},
            ],
        },
        {
            "task": "MR",
            "id": 3,
            "history": [{"user": "q", "bot": "ref"}],
        },
    ]
    eval_items = expand_mt_bench_101_records(records)
    assert len(eval_items) == 4
    cm_rows = eval_items[eval_items["task"] == "CM"]
    assert cm_rows.iloc[0]["turn_index"] == 2
    assert len(cm_rows.iloc[0]["golden_context"]) == 1
    mr_rows = eval_items[eval_items["task"] == "MR"]
    assert mr_rows.iloc[0]["requires_reference"]
    assert mr_rows.iloc[0]["reference_answer"] == "ref"
    assert mr_rows.iloc[0]["ability"] == "reasoning"
    assert mr_rows.iloc[0]["domain"] == "adaptability"


def test_expand_mt_bench_101_unknown_task():
    with pytest.raises(ValueError, match="Unknown MT-Bench-101 task"):
        expand_mt_bench_101_records([{"task": "XX", "id": 1, "history": []}])


def test_download_mt_bench_101_uses_pinned_revision(tmp_path, monkeypatch):
    task = get_packaged_task("mt-bench-101")
    assert task is not None
    calls = {}

    def _urlretrieve(url, dest):
        calls["url"] = url
        dest.write_text("{}\n")

    monkeypatch.setattr(mt_bench_101, "urlretrieve", _urlretrieve)
    mt_bench_101.download_task_sources(task, tmp_path)
    assert "bc18b3e2c18c99164e11528f1a79c92083db5953" in calls["url"]
    assert calls["url"].endswith("data/subjective/mtbench101.jsonl")


def test_load_task_instructions_expands_cached_jsonl(tmp_path, monkeypatch):
    task = get_packaged_task("mt-bench-101")
    assert task is not None
    dataset_path = tmp_path / "_sources" / "mt-bench-101" / "mtbench101.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        json.dumps(
            {
                "task": "PI",
                "id": 9,
                "history": [{"user": "hello", "bot": "world"}],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(mt_bench_101, "download_task_sources", lambda *_args: None)
    loaded = mt_bench_101.load_task_instructions(task, tmp_path)
    assert loaded.iloc[0]["dialogue_uid"] == "PI:9"
    assert loaded.iloc[0]["instruction"] == "hello"
