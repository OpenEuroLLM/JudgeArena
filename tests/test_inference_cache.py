import json
import sys
from types import SimpleNamespace

import pytest

from judgearena.inference_cache import InferenceCache
from judgearena.models import do_inference, make_model
from judgearena.store_sqlite import (
    SQLiteInferenceStore,
    descriptor_hash,
    store_folder,
)


def _install_fake_vllm(monkeypatch):
    captured = {"llm_init": False}
    monkeypatch.setattr(
        "judgearena.model_adapters._provider_package_version",
        lambda name: "test-version",
    )

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_kwargs"] = kwargs

    class FakeReasoningConfig:
        def __init__(self, **kwargs):
            captured["reasoning_config_kwargs"] = kwargs

    class FakeLLM:
        def __init__(self, *, model, trust_remote_code, **kwargs):
            captured["llm_init"] = True
            captured["llm_init_args"] = {
                "model": model,
                "trust_remote_code": trust_remote_code,
                "kwargs": kwargs,
            }

        def get_tokenizer(self):
            return SimpleNamespace(chat_template="{{ messages }}")

        def chat(self, messages, sampling_params, **kwargs):
            return [SimpleNamespace(outputs=[SimpleNamespace(text="generated")])]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.config.reasoning",
        SimpleNamespace(ReasoningConfig=FakeReasoningConfig),
    )

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model, trust_remote_code=True):
            return SimpleNamespace(chat_template="{{ messages }}")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model, trust_remote_code=True):
            return SimpleNamespace(max_position_embeddings=8192)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeAutoTokenizer,
            AutoConfig=FakeAutoConfig,
        ),
    )
    return captured


def _seed_cache(tmp_path, task, model, inputs, outputs):
    descriptor = model.cache_descriptor()
    assert descriptor is not None
    canonical = [model.canonicalize_input(item) for item in inputs]
    with InferenceCache(tmp_path, task, mode="refresh") as cache:
        cache.get_or_run(
            model_spec=model.model_spec,
            descriptor=descriptor,
            canonical_inputs=canonical,
            original_inputs=inputs,
            miss_runner=lambda miss_inputs: outputs[: len(miss_inputs)],
            producer_metadata=model.producer_metadata(),
        )


def test_vllm_full_cache_hit_skips_llm_init(monkeypatch, tmp_path):
    captured = _install_fake_vllm(monkeypatch)
    model = make_model("VLLM/Qwen/Qwen3.5-9B", max_tokens=16, temperature=0.0)
    inputs = ["hello", "world"]
    _seed_cache(tmp_path, "arena", model, inputs, ["cached-a", "cached-b"])

    hit_model = make_model("VLLM/Qwen/Qwen3.5-9B", max_tokens=16, temperature=0.0)
    with InferenceCache(tmp_path, "arena", mode="use") as cache:
        results = do_inference(hit_model, inputs, cache=cache)

    assert results == ["cached-a", "cached-b"]
    assert captured["llm_init"] is False


def test_within_batch_dedupe_runs_inference_once(tmp_path):
    model = make_model("Dummy/cache-dedupe", max_tokens=8)
    inputs = ["same", "same", "other"]
    seen: list[str] = []

    def miss_runner(miss_inputs):
        seen.extend(miss_inputs)
        return [f"out-{item}" for item in miss_inputs]

    descriptor = model.cache_descriptor()
    canonical = [model.canonicalize_input(item) for item in inputs]
    with InferenceCache(tmp_path, "arena", mode="refresh") as cache:
        first = cache.get_or_run(
            model_spec=model.model_spec,
            descriptor=descriptor,
            canonical_inputs=canonical,
            original_inputs=inputs,
            miss_runner=miss_runner,
            producer_metadata=model.producer_metadata(),
        )

    assert first == ["out-same", "out-same", "out-other"]
    assert seen == ["same", "other"]

    seen.clear()
    with InferenceCache(tmp_path, "arena", mode="use") as cache:
        second = do_inference(model, inputs, cache=cache)

    assert second == first
    assert seen == []


def test_use_mode_returns_row_won_by_concurrent_insert(monkeypatch, tmp_path):
    model = make_model("Dummy/concurrent", max_tokens=8)
    descriptor = model.cache_descriptor()
    assert descriptor is not None
    canonical = [model.canonicalize_input("input")]
    original_save = SQLiteInferenceStore.save_outputs_and_metadata

    def racing_save(self, outputs, metadata, **kwargs):
        competing = outputs.copy()
        competing["output_text"] = "concurrent-winner"
        self.save_outputs(competing, pushed_by="other", run_id="other")
        return original_save(self, outputs, metadata, **kwargs)

    monkeypatch.setattr(
        SQLiteInferenceStore,
        "save_outputs_and_metadata",
        racing_save,
    )
    with InferenceCache(tmp_path, "arena", mode="use") as cache:
        result = cache.get_or_run(
            model_spec=model.model_spec,
            descriptor=descriptor,
            canonical_inputs=canonical,
            original_inputs=["input"],
            miss_runner=lambda _: ["fresh-loser"],
            row_metadata=[{"question_id": "q1"}],
        )

    assert result == ["concurrent-winner"]


def test_changed_input_is_cache_miss(tmp_path):
    model = make_model("Dummy/cache-change", max_tokens=8)
    _seed_cache(tmp_path, "arena", model, ["alpha"], ["out-alpha"])

    with InferenceCache(tmp_path, "arena", mode="use") as cache:
        results = do_inference(model, ["beta"], cache=cache)

    assert results == ["cache-change"]


def test_seed_separates_descriptor_cells(tmp_path):
    model_a = make_model("Dummy/cache-seed", max_tokens=8, seed=1)
    model_b = make_model("Dummy/cache-seed", max_tokens=8, seed=2)
    assert model_a.cache_descriptor() != model_b.cache_descriptor()

    _seed_cache(tmp_path, "arena", model_a, ["prompt"], ["seed-1"])
    with InferenceCache(tmp_path, "arena", mode="use") as cache:
        hit_a = do_inference(model_a, ["prompt"], cache=cache)
        miss_b = do_inference(model_b, ["prompt"], cache=cache)

    assert hit_a == ["seed-1"]
    assert miss_b == ["cache-seed"]


def test_refresh_replaces_cached_output(tmp_path):
    model = make_model("Dummy/cache-refresh", max_tokens=8)
    _seed_cache(tmp_path, "arena", model, ["prompt"], ["old"])

    with InferenceCache(tmp_path, "arena", mode="refresh") as cache:
        results = do_inference(model, ["prompt"], cache=cache)

    assert results == ["cache-refresh"]


def test_metadata_associations_are_saved(tmp_path):
    model = make_model("Dummy/cache-meta", max_tokens=8)
    metadata = [{"question_id": "q-1"}, {"question_id": "q-2"}]
    with InferenceCache(tmp_path, "arena", mode="refresh") as cache:
        do_inference(
            model,
            ["one", "two"],
            cache=cache,
            cache_meta={"metadata": metadata},
        )

    descriptor = model.cache_descriptor()
    folder = store_folder(
        tmp_path,
        "arena",
        model.model_spec,
        descriptor_hash(descriptor),
    )

    with SQLiteInferenceStore(folder / "inference.db") as store:
        rows = store.query_metadata()

    assert len(rows) == 2
    saved = {json.loads(value)["question_id"] for value in rows["metadata_json"]}
    assert saved == {"q-1", "q-2"}


def test_off_mode_never_reads_cache(tmp_path):
    model = make_model("Dummy/off", max_tokens=8)
    _seed_cache(tmp_path, "arena", model, ["prompt"], ["cached"])

    with InferenceCache(tmp_path, "arena", mode="off") as cache:
        results = do_inference(model, ["prompt"], cache=cache)

    assert results == ["off"]


def test_off_mode_runs_every_input_without_dedupe(tmp_path):
    model = make_model("Dummy/off-dedupe", max_tokens=8)
    calls: list[str] = []

    def counting_runner(inputs):
        calls.extend(inputs)
        return [f"out-{index}" for index, _ in enumerate(inputs)]

    with InferenceCache(tmp_path, "arena", mode="off") as cache:
        results = cache.get_or_run(
            model_spec=model.model_spec,
            descriptor=model.cache_descriptor(),
            canonical_inputs=[model.canonicalize_input("same")] * 2,
            original_inputs=["same", "same"],
            miss_runner=counting_runner,
            producer_metadata=model.producer_metadata(),
        )

    assert results == ["out-0", "out-1"]
    assert calls == ["same", "same"]


def test_close_before_push_closes_sqlite_before_upload(monkeypatch, tmp_path):
    model = make_model("Dummy/push-close", max_tokens=8)
    states: list[bool] = []
    active_caches: list[InferenceCache] = []

    import judgearena.inference_cache as inference_cache_mod

    original_enter = InferenceCache.__enter__

    def track_enter(self):
        active_caches.append(self)
        return original_enter(self)

    monkeypatch.setattr(InferenceCache, "__enter__", track_enter)

    def spy_push(*args, **kwargs):
        cache = active_caches[-1]
        states.append(bool(cache._stores))
        return None

    monkeypatch.setattr(inference_cache_mod, "push_cells", spy_push)

    with InferenceCache(tmp_path, "arena", mode="refresh", push=True) as cache:
        do_inference(model, ["x"], cache=cache)

    assert states == [False]


def test_invalid_cache_mode_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid cache mode"):
        InferenceCache(tmp_path, "arena", mode="bogus")  # type: ignore[arg-type]
