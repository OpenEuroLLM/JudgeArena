import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

import judgearena.inference as inference
import judgearena.models as models
from judgearena.inference import (
    CompletionInferenceCache,
    JudgementInferenceCache,
    canonicalize_model_input,
    provider_input_mode,
)
from judgearena.models import InferenceResult, do_inference, prepare_model
from judgearena.usage import track_usage


class EchoModel:
    def __init__(self):
        self.calls = []

    def batch(self, inputs, **_kwargs):
        self.calls.append(inputs)
        return [AIMessage(content=f"generated:{item}") for item in inputs]


def test_full_hit_does_not_materialize_model(tmp_path, monkeypatch):
    cache = CompletionInferenceCache(tmp_path, "arena-hard")
    monkeypatch.setattr(models, "make_model", lambda *_args, **_kwargs: EchoModel())
    metadata = [{"instruction_id": "1"}]
    do_inference(
        prepare_model("Dummy/test-model", cache=cache),
        ["prompt"],
        cache_metadata=metadata,
    )

    def fail_if_materialized(*_args, **_kwargs):
        raise AssertionError("cache hit materialized the model")

    monkeypatch.setattr(models, "make_model", fail_if_materialized)
    with track_usage() as tracker:
        outputs = do_inference(
            prepare_model("Dummy/test-model", cache=cache),
            ["prompt"],
            cache_metadata=metadata,
        )

    assert outputs == ["generated:prompt"]
    assert tracker.snapshot().requests == ()


def test_mixed_hits_and_misses_preserve_order(tmp_path, monkeypatch):
    cache = CompletionInferenceCache(tmp_path, "arena-hard")
    first_backend = EchoModel()
    monkeypatch.setattr(models, "make_model", lambda *_args, **_kwargs: first_backend)
    do_inference(
        prepare_model("Dummy/test-model", cache=cache),
        ["hit"],
        cache_metadata=[{"instruction_id": "hit"}],
    )

    backend = EchoModel()
    monkeypatch.setattr(models, "make_model", lambda *_args, **_kwargs: backend)
    outputs = do_inference(
        prepare_model("Dummy/test-model", cache=cache),
        ["miss-a", "hit", "miss-b"],
        cache_metadata=[
            {"instruction_id": "a"},
            {"instruction_id": "hit"},
            {"instruction_id": "b"},
        ],
    )

    assert outputs == ["generated:miss-a", "generated:hit", "generated:miss-b"]
    assert backend.calls == [["miss-a", "miss-b"]]


def test_judgement_hit_preserves_top_logprobs(tmp_path, monkeypatch):
    cache = JudgementInferenceCache(tmp_path, "arena-hard")

    class LogprobModel:
        def batch(self, inputs, **_kwargs):
            return [
                InferenceResult(
                    text="m",
                    first_token_top_logprobs={"m": -0.1, "M": -2.0},
                )
                for _ in inputs
            ]

    monkeypatch.setattr(models, "make_model", lambda *_args, **_kwargs: LogprobModel())
    metadata = [
        {
            "instruction_id": "1",
            "model_a": "candidate",
            "model_b": "baseline",
            "orientation": "direct",
        }
    ]
    first = do_inference(
        prepare_model("Dummy/judge", cache=cache),
        ["judge prompt"],
        return_top_logprobs=True,
        cache_metadata=metadata,
    )
    second = do_inference(
        prepare_model("Dummy/judge", cache=cache),
        ["judge prompt"],
        return_top_logprobs=True,
        cache_metadata=metadata,
    )

    assert second[0].text == first[0].text
    assert second[0].first_token_top_logprobs == first[0].first_token_top_logprobs
    assert second[0].usage is None


def test_vllm_descriptor_contains_output_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(inference.importlib_metadata, "version", lambda _name: "0.10.2")
    cache = CompletionInferenceCache(tmp_path, "arena-hard")

    descriptor = prepare_model(
        "VLLM/Qwen/Qwen3-8B",
        max_tokens=32,
        cache=cache,
        temperature=0.2,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        tensor_parallel_size=2,
    ).descriptor

    assert descriptor["backend_version"] == "0.10.2"
    assert descriptor["model_kwargs"]["temperature"] == 0.2
    assert descriptor["model_kwargs"]["top_p"] == 0.95
    assert descriptor["model_kwargs"]["max_model_len"] == 4096
    assert "tensor_parallel_size" not in descriptor["model_kwargs"]


@pytest.mark.parametrize(
    ("model_spec", "expected_mode", "expected_endpoint"),
    [
        ("Dummy/model", "chat", None),
        ("VLLM/org/model", "chat", None),
        ("OpenRouter/org/model", "chat", "https://openrouter.ai/api/v1"),
        ("ChatOpenAI/model", "chat", "https://api.openai.com/v1"),
        ("OpenAI/model", "text", "https://api.openai.com/v1"),
        ("Together/org/model", "text", "https://api.together.xyz/v1/completions"),
        ("LlamaCpp/./models/model.gguf", "text", None),
    ],
)
def test_supported_provider_descriptors(
    tmp_path,
    monkeypatch,
    model_spec,
    expected_mode,
    expected_endpoint,
):
    versions = {"vllm": "0.10.2", "llama-cpp-python": "0.3.0"}
    monkeypatch.setattr(inference.importlib_metadata, "version", versions.__getitem__)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    cache = CompletionInferenceCache(tmp_path, "arena-hard")

    descriptor = prepare_model(model_spec, cache=cache).descriptor

    assert descriptor["input_mode"] == expected_mode
    assert descriptor.get("endpoint") == expected_endpoint


def test_hosted_descriptor_hashes_routing_and_endpoint(tmp_path, monkeypatch, caplog):
    cache = CompletionInferenceCache(tmp_path, "arena-hard")
    unpinned = prepare_model("OpenRouter/org/model", cache=cache).descriptor
    assert "uses unpinned provider routing" in caplog.text

    caplog.clear()
    pinned = prepare_model(
        "OpenRouter/org/model",
        cache=cache,
        extra_body={"provider": {"order": ["Together"], "allow_fallbacks": False}},
    ).descriptor
    assert "uses unpinned provider routing" not in caplog.text
    assert unpinned != pinned

    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example/v1")
    ambient = prepare_model("ChatOpenAI/model", cache=cache).descriptor
    assert ambient["endpoint"] == "https://ambient.example/v1"


def test_unsupported_provider_runs_uncached(tmp_path, caplog):
    model = prepare_model(
        "Unsupported/model",
        cache=CompletionInferenceCache(tmp_path, "arena-hard"),
    )

    assert model.descriptor is None
    assert "Caching is not supported" in caplog.text


@pytest.mark.parametrize(
    ("provider", "expected_type"),
    [("OpenRouter", "messages"), ("Together", "text"), ("VLLM", "messages")],
)
def test_input_canonicalization_matches_provider_mode(provider, expected_type):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "System"), ("user", "Question")]
    ).invoke({})

    payload = json.loads(
        canonicalize_model_input(prompt, provider_input_mode(provider))
    )

    assert payload["type"] == expected_type
