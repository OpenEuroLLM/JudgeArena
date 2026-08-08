import json

import pandas as pd
import pytest

from judgearena.cache_sqlite import (
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
        cache.save(first)
        cache.save(second)
        result = cache.query([input_hash("rendered prompt")])

    assert result["completion"].tolist() == ["second"]


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
        cache.save(rows)
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
                "orientation": "direct",
            }
            for index in range(2)
        ]
    )

    with JudgementCache(tmp_path / JUDGEMENT_DB_NAME) as cache:
        cache.save(rows)
        result = cache.query(model="candidate")
        assert result["judge_completion"].tolist() == ["scores 0"]
        assert cache.delete(model="candidate") == 1
        assert cache.query()["instruction_id"].tolist() == ["1"]
