import json

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
