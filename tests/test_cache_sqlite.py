import json
import sqlite3

import pandas as pd
import pytest

from judgearena.cache.sqlite import (
    COMPLETION_DB_NAME,
    JUDGEMENT_DB_NAME,
    CompletionCache,
    JudgementCache,
    cache_folder,
    input_hash,
    write_descriptor,
)
from judgearena.usage import RequestUsage

DESCRIPTOR = {
    "model": "Qwen/Qwen3-8B",
    "provider": "VLLM",
    "sampling": {"max_tokens": 1024, "temperature": 0.0},
}


def test_descriptor_paths_separate_completion_and_judgement_caches(tmp_path):
    model = "VLLM/Qwen/Qwen3-8B"
    completion_folder = cache_folder(
        tmp_path, "completions", "arena-hard", model, DESCRIPTOR
    )
    judgement_folder = cache_folder(
        tmp_path, "judgements", "arena-hard", model, DESCRIPTOR
    )

    assert completion_folder != judgement_folder
    assert completion_folder.name == judgement_folder.name
    assert completion_folder / COMPLETION_DB_NAME != (
        judgement_folder / JUDGEMENT_DB_NAME
    )

    metadata_path = write_descriptor(completion_folder, DESCRIPTOR)
    assert json.loads(metadata_path.read_text()) == DESCRIPTOR
    write_descriptor(completion_folder, DESCRIPTOR)

    with pytest.raises(ValueError, match="does not match"):
        write_descriptor(completion_folder, {**DESCRIPTOR, "provider": "OpenAI"})


def test_completion_cache_uses_content_key_and_last_write(tmp_path):
    db_path = tmp_path / COMPLETION_DB_NAME
    first = pd.DataFrame(
        [
            {
                "input_text": "rendered prompt",
                "completion": "first",
                "benchmark": "arena-hard",
                "instruction_id": "12",
                "model": "VLLM/Qwen/Qwen3-8B",
            }
        ]
    )
    second = first.assign(completion="second")

    with CompletionCache(db_path) as cache:
        cache.save(first, pushed_by="alice")
        cache.save(second, pushed_by="bob")
        result = cache.query([input_hash("rendered prompt")])
        assert cache.query([]).empty
        assert len(cache.query(None)) == 1

    assert result["completion"].tolist() == ["second"]
    assert result["pushed_by"].tolist() == ["bob"]


def test_completion_cache_filters_and_deletes_by_instruction(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "input_text": f"prompt-{index}",
                "completion": f"completion-{index}",
                "benchmark": "arena-hard",
                "instruction_id": str(index),
                "model": "VLLM/Qwen/Qwen3-8B",
            }
            for index in range(2)
        ]
    )

    with CompletionCache(tmp_path / COMPLETION_DB_NAME) as cache:
        cache.save(rows, pushed_by="alice")
        assert cache.query(instruction_id="1")["completion"].tolist() == [
            "completion-1"
        ]
        assert cache.delete(instruction_id="1") == 1
        assert cache.query()["instruction_id"].tolist() == ["0"]


def test_judgement_cache_filters_and_deletes_by_candidate_model(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "judge_input": f"judge prompt {index}",
                "judge_completion": f"scores {index}",
                "benchmark": "arena-hard",
                "instruction_id": str(index),
                "model_a": "candidate" if index == 0 else "baseline",
                "model_b": "baseline" if index == 0 else "other",
                "judge": "VLLM/Qwen/Qwen3-8B",
                "top_logprobs": {"m": -0.1, "M": -2.0} if index == 0 else None,
                "orientation": "direct",
            }
            for index in range(2)
        ]
    )

    with JudgementCache(tmp_path / JUDGEMENT_DB_NAME) as cache:
        cache.save(rows, pushed_by="alice")
        result = cache.query(model="candidate")
        assert result["judge_completion"].tolist() == ["scores 0"]
        assert json.loads(result.iloc[0]["top_logprobs"]) == {"M": -2.0, "m": -0.1}
        assert cache.delete(model="candidate") == 1
        assert cache.query()["instruction_id"].tolist() == ["1"]


def test_judgement_cache_preserves_null_model_b(tmp_path):
    row = pd.DataFrame(
        [
            {
                "judge_input": "pointwise prompt",
                "judge_completion": "Rating: [[8]]",
                "benchmark": "mt-bench-101",
                "instruction_id": "1",
                "model_a": "candidate",
                "model_b": None,
                "judge": "VLLM/Qwen/Qwen3-8B",
                "top_logprobs": None,
                "orientation": "single",
            }
        ]
    )

    with JudgementCache(tmp_path / JUDGEMENT_DB_NAME) as cache:
        cache.save(row, pushed_by="alice")
        assert cache.query().iloc[0]["model_b"] is None


def test_merge_from_updates_live_database_in_place(tmp_path):
    local_path = tmp_path / "local" / COMPLETION_DB_NAME
    incoming_path = tmp_path / "incoming" / COMPLETION_DB_NAME
    shared = pd.DataFrame(
        [
            {
                "input_text": "shared",
                "completion": "local",
                "benchmark": "arena-hard",
                "instruction_id": "1",
                "model": "VLLM/Qwen/Qwen3-8B",
            }
        ]
    )
    with CompletionCache(incoming_path) as incoming:
        incoming.save(
            pd.concat(
                [
                    shared.assign(completion="incoming"),
                    shared.assign(
                        input_text="new",
                        completion="new",
                        instruction_id="2",
                    ),
                ],
                ignore_index=True,
            ),
            pushed_by="bob",
        )
        incoming._connect().execute(
            "UPDATE completions SET pushed_at = '2030-01-01T00:00:00+00:00'"
        )
        incoming._connect().commit()

    with CompletionCache(local_path) as cache:
        cache.save(shared, pushed_by="alice")
        inode = local_path.stat().st_ino
        assert cache.merge_from(incoming_path) == 2
        assert local_path.stat().st_ino == inode
        assert cache.query()["completion"].tolist() == ["incoming", "new"]


def test_completion_usage_round_trips_and_legacy_database_upgrades(tmp_path):
    db_path = tmp_path / COMPLETION_DB_NAME
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """CREATE TABLE completions (
            input_hash TEXT PRIMARY KEY, input_text TEXT NOT NULL,
            completion TEXT NOT NULL, benchmark TEXT NOT NULL,
            instruction_id TEXT NOT NULL, model TEXT NOT NULL,
            pushed_by TEXT NOT NULL, pushed_at TEXT NOT NULL, run_id TEXT NOT NULL
        )"""
    )
    legacy.commit()
    legacy.close()

    rows = pd.DataFrame(
        [
            {
                "input_text": "prompt",
                "completion": "answer",
                "benchmark": "arena",
                "instruction_id": "1",
                "model": "Dummy/model",
                "usage_json": RequestUsage(
                    stage="generation",
                    input_tokens=3,
                    output_tokens=5,
                    cached_tokens=1,
                    reasoning_tokens=2,
                    total_tokens=8,
                    model="Dummy/model",
                    cost_usd=0.25,
                ),
            }
        ]
    )
    with CompletionCache(db_path) as cache:
        cache.save(rows, pushed_by="alice")
        result = cache.query().iloc[0]

    assert "usage_json" in result.index
    assert json.loads(result["usage_json"]) == {
        "cached_tokens": 1,
        "input_tokens": 3,
        "model": "Dummy/model",
        "output_tokens": 5,
        "reasoning_tokens": 2,
        "stage": "generation",
        "total_tokens": 8,
    }


@pytest.mark.parametrize(
    ("incoming_completion", "expect_prior_usage"),
    [("old answer", False), ("native answer", True)],
)
def test_merge_legacy_incoming_database_projects_null_usage(
    tmp_path, incoming_completion, expect_prior_usage
):
    local_path = tmp_path / "local" / COMPLETION_DB_NAME
    incoming_path = tmp_path / "legacy" / COMPLETION_DB_NAME
    incoming_path.parent.mkdir(parents=True)
    legacy = sqlite3.connect(incoming_path)
    legacy.execute(
        """CREATE TABLE completions (
            input_hash TEXT PRIMARY KEY, input_text TEXT NOT NULL,
            completion TEXT NOT NULL, benchmark TEXT NOT NULL,
            instruction_id TEXT NOT NULL, model TEXT NOT NULL,
            pushed_by TEXT NOT NULL, pushed_at TEXT NOT NULL, run_id TEXT NOT NULL
        )"""
    )
    legacy.execute(
        "INSERT INTO completions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            input_hash("old prompt"),
            "old prompt",
            incoming_completion,
            "arena",
            "1",
            "Dummy/model",
            "bob",
            "2030-01-01T00:00:00+00:00",
            "old-run",
        ),
    )
    legacy.commit()
    legacy.close()

    prior_usage = RequestUsage(stage="generation", input_tokens=4)
    with CompletionCache(local_path) as cache:
        cache.save(
            pd.DataFrame(
                [
                    {
                        "input_text": "old prompt",
                        "completion": "native answer",
                        "benchmark": "arena",
                        "instruction_id": "1",
                        "model": "Dummy/model",
                        "usage_json": prior_usage,
                    }
                ]
            ),
            pushed_by="alice",
        )
        assert cache.merge_from(incoming_path) == 1
        result = cache.query().iloc[0]

    assert result["completion"] == incoming_completion
    if expect_prior_usage:
        assert json.loads(result["usage_json"])["input_tokens"] == 4
    else:
        assert pd.isna(result["usage_json"])

    incoming = sqlite3.connect(incoming_path)
    incoming_columns = {
        column[1] for column in incoming.execute("PRAGMA table_info(completions)")
    }
    incoming.close()
    assert "usage_json" not in incoming_columns
