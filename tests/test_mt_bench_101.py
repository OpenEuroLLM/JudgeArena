import json

import pandas as pd
import pytest

import judgearena.benchmarks.mt_bench_101.runner as runner
import judgearena.datasets.mt_bench_101 as mt_bench_101
from judgearena.benchmarks.mt_bench_101.evaluate import (
    derive_mt_bench_101_pairwise_preferences,
    judge_mt_bench_101_single,
    parse_mt_bench_101_rating,
    summarize_mt_bench_101_absolute_scores,
)
from judgearena.benchmarks.mt_bench_101.runner import run_mt_bench_101_benchmark
from judgearena.config import RunConfig
from judgearena.datasets.mt_bench_101 import expand_mt_bench_101_records
from judgearena.models import DummyModel
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


def test_parse_mt_bench_101_rating():
    assert parse_mt_bench_101_rating("Reasoning...\nRating: [[7]]") == pytest.approx(
        7.0
    )
    assert parse_mt_bench_101_rating("Rating: [[0]]") is None
    assert parse_mt_bench_101_rating("Rating: [6]") is None


def test_judge_mt_bench_101_includes_reference_block_for_mr():
    eval_items = pd.DataFrame(
        {
            "instruction_index": [0],
            "dialogue_id": [1],
            "dialogue_uid": ["MR:1"],
            "task": ["MR"],
            "ability": ["adaptability"],
            "turn_index": [2],
            "golden_context": [[{"user": "q1", "bot": "a1"}]],
            "user_message": ["q2"],
            "reference_answer": ["ref answer"],
        }
    ).set_index("instruction_index")
    completions = pd.DataFrame(
        {"instruction_index": [0], "completion": ["model answer"]}
    )
    scored = judge_mt_bench_101_single(
        judge_chat_model=DummyModel("Dummy/Rating: [[8]]"),
        eval_items=eval_items,
        completions=completions,
        use_tqdm=False,
    )
    user_prompt = scored.iloc[0]["user_prompt"]
    assert scored.iloc[0]["score"] == pytest.approx(8.0)
    assert "The reference solution is:" in user_prompt
    assert "ref answer" in user_prompt


def test_mt_bench_101_min_dialogue_and_pairwise():
    scored_a = pd.DataFrame(
        {
            "instruction_index": [0, 1, 2],
            "dialogue_uid": ["PI:1", "PI:1", "PI:2"],
            "dialogue_id": [1, 1, 2],
            "task": ["PI", "PI", "PI"],
            "ability": ["interactivity", "interactivity", "interactivity"],
            "turn_index": [1, 2, 1],
            "score": [9.0, 2.0, 4.0],
        }
    )
    scored_b = scored_a.assign(score=[8.0, 1.0, 6.0])
    absolute_a = summarize_mt_bench_101_absolute_scores(scored_a)
    assert absolute_a["per_task"]["PI"] == pytest.approx(3.0)
    pairwise = derive_mt_bench_101_pairwise_preferences(scored_a, scored_b)
    assert len(pairwise) == 3


def test_run_mt_bench_101_dummy(tmp_path, monkeypatch):
    eval_items = expand_mt_bench_101_records(
        [{"task": "PI", "id": 1, "history": [{"user": "q", "bot": "a"}]}]
    ).set_index("instruction_index")
    monkeypatch.setattr(runner, "load_instructions", lambda *_a, **_k: eval_items)
    monkeypatch.setattr(runner, "cache_function_dataframe", lambda fun, **_k: fun())
    cfg = RunConfig(
        task="mt-bench-101",
        model={"name": "Dummy/ok-a", "baseline": "Dummy/ok-b"},
        judge={"model": "Dummy/Rating: [[8]]"},
        generation={"n_instructions": 1},
        run={"result_folder": str(tmp_path), "ignore_cache": True, "use_tqdm": False},
    )
    prefs = run_mt_bench_101_benchmark(cfg, get_packaged_task("mt-bench-101"))
    assert len(prefs) == 1
    assert prefs.iloc[0] == pytest.approx(0.5)
