from __future__ import annotations

import json

import pandas as pd

import judgearena.generate as generate_module
from judgearena.generate import generate_base, generate_multiturn
from judgearena.inference_cache import InferenceCache
from judgearena.models import make_model
from judgearena.store_sqlite import SQLiteInferenceStore, descriptor_hash, store_folder


def test_generate_base_routes_through_do_inference(monkeypatch):
    calls: list[dict] = []
    real_do_inference = generate_module.do_inference

    def spy_do_inference(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return real_do_inference(*args, **kwargs)

    monkeypatch.setattr(generate_module, "do_inference", spy_do_inference)

    instructions = pd.Series(["hello", "world"], index=[10, 20])
    df = generate_base(instructions, "Dummy/generate-base-path", use_tqdm=False)

    assert len(calls) == 1
    assert calls[0]["kwargs"]["use_tqdm"] is False
    assert calls[0]["kwargs"]["cache"] is None
    assert df["completion"].tolist() == ["generate-base-path"] * 2
    assert df["instruction_index"].tolist() == [10, 20]


def test_generate_base_forwards_cache_and_metadata(monkeypatch):
    captured: list[dict] = []

    def spy_do_inference(*, cache, cache_meta, **kwargs):
        captured.append({"cache": cache, "cache_meta": cache_meta})
        return ["out-a", "out-b"]

    monkeypatch.setattr(generate_module, "do_inference", spy_do_inference)

    instructions = pd.Series(["a", "b"], index=["q1", "q2"])
    with InferenceCache("/tmp/unused", "gen-task", mode="off") as cache:
        df = generate_base(
            instructions,
            "Dummy/ignored",
            cache=cache,
        )

    assert df["completion"].tolist() == ["out-a", "out-b"]
    assert captured[0]["cache"] is cache
    assert captured[0]["cache_meta"] == {
        "metadata": [
            {"instruction_index": "q1"},
            {"instruction_index": "q2"},
        ]
    }


def test_generate_base_cache_hit_skips_backend_batch(tmp_path):
    model = make_model("Dummy/cache-generate-base", max_tokens=8)
    inputs = ["alpha", "beta"]
    descriptor = model.cache_descriptor()
    assert descriptor is not None
    canonical = [model.canonicalize_input(item) for item in inputs]
    metadata = [{"instruction_index": "0"}, {"instruction_index": "1"}]

    with InferenceCache(tmp_path, "gen", mode="refresh") as cache:
        cache.get_or_run(
            model_spec=model.model_spec,
            descriptor=descriptor,
            canonical_inputs=canonical,
            original_inputs=inputs,
            miss_runner=lambda miss_inputs: [f"cached-{item}" for item in miss_inputs],
            row_metadata=metadata,
            producer_metadata=model.producer_metadata(),
        )

    instructions = pd.Series(inputs, index=[0, 1])
    with InferenceCache(tmp_path, "gen", mode="use") as cache:
        df = generate_base(
            instructions,
            "Dummy/cache-generate-base",
            max_tokens=8,
            cache=cache,
        )

    assert df["completion"].tolist() == ["cached-alpha", "cached-beta"]


def test_generate_multiturn_metadata_and_temperature_groups(monkeypatch):
    calls: list[dict] = []

    def spy_do_inference(*, inputs, cache_meta=None, **kwargs):
        calls.append({"inputs": inputs, "cache_meta": cache_meta})
        return [f"out-{index}" for index in range(len(inputs))]

    monkeypatch.setattr(generate_module, "do_inference", spy_do_inference)

    questions = pd.DataFrame(
        {
            "category": ["writing", "math", "writing"],
            "turn_1": ["Q1", "Q2", "Q3"],
            "turn_2": ["Q1b", "Q2b", "Q3b"],
        },
        index=pd.Index([1, 2, 3], name="instruction_index"),
    )
    temperature_config = {"writing": 0.5, "math": 0.9}

    df = generate_multiturn(
        questions,
        "Dummy/multiturn",
        temperature_config=temperature_config,
        use_tqdm=False,
    )

    assert len(df) == 3
    assert len(calls) == 4

    turn1_calls = calls[:2]
    turn2_calls = calls[2:]

    assert turn1_calls[0]["cache_meta"]["metadata"] == [
        {"instruction_index": "1", "turn": 1, "category": "writing"},
        {"instruction_index": "3", "turn": 1, "category": "writing"},
    ]
    assert turn1_calls[1]["cache_meta"]["metadata"] == [
        {"instruction_index": "2", "turn": 1, "category": "math"},
    ]
    assert len(turn1_calls[0]["inputs"]) == 2
    assert len(turn1_calls[1]["inputs"]) == 1

    assert turn2_calls[0]["cache_meta"]["metadata"] == [
        {"instruction_index": "1", "turn": 2, "category": "writing"},
        {"instruction_index": "3", "turn": 2, "category": "writing"},
    ]
    assert turn2_calls[1]["cache_meta"]["metadata"] == [
        {"instruction_index": "2", "turn": 2, "category": "math"},
    ]


def test_generate_multiturn_saves_metadata_in_cache(tmp_path):
    questions = pd.DataFrame(
        {
            "category": ["writing"],
            "turn_1": ["Q1"],
            "turn_2": ["Q2"],
        },
        index=pd.Index([7], name="instruction_index"),
    )

    with InferenceCache(tmp_path, "mt-bench", mode="refresh") as cache:
        generate_multiturn(
            questions,
            "Dummy/mt-meta",
            use_tqdm=False,
            cache=cache,
        )

    model = make_model("Dummy/mt-meta", max_tokens=8192)
    descriptor = model.cache_descriptor()
    folder = store_folder(
        tmp_path,
        "mt-bench",
        model.model_spec,
        descriptor_hash(descriptor),
    )
    with SQLiteInferenceStore(folder / "inference.db") as store:
        rows = store.query_metadata()

    saved = [json.loads(value) for value in rows["metadata_json"]]
    assert {"instruction_index": "7", "turn": 1, "category": "writing"} in saved
    assert {"instruction_index": "7", "turn": 2, "category": "writing"} in saved
