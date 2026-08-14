import json
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.prompts import ChatPromptTemplate

import judgearena.evaluate as evaluate
import judgearena.generate as generate
import judgearena.inference as inference
import judgearena.utils as utils
from judgearena.inference import (
    CompletionInferenceCache,
    JudgementInferenceCache,
    canonicalize_model_input,
    provider_input_mode,
)
from judgearena.utils import do_inference, prepare_model


class ConstantModel:
    def __init__(self, output):
        self.output = output

    def batch(self, inputs, **_kwargs):
        return [self.output] * len(inputs)


class EchoModel:
    def __init__(self):
        self.calls = []

    def batch(self, inputs, **_kwargs):
        self.calls.append(inputs)
        return [f"generated:{item}" for item in inputs]


def test_legacy_runtime_cache_symbols_are_removed():
    package = Path(utils.__file__).parent
    source = "\n".join(path.read_text() for path in package.rglob("*.py"))
    for symbol in ("cache_function_dataframe", "ignore_cache", "set_langchain_cache"):
        assert symbol not in source


def test_generate_and_judge_full_hits_do_not_materialize_models(tmp_path, monkeypatch):
    completion_cache = CompletionInferenceCache(tmp_path, "arena-hard")
    judgement_cache = JudgementInferenceCache(tmp_path, "arena-hard")
    battle_model = "VLLM/Qwen/Qwen3-8B"
    judge_model = "VLLM/Qwen/Qwen3-32B"
    monkeypatch.setattr(inference.importlib_metadata, "version", lambda _name: "0.10.2")

    def fake_model(model, **_kwargs):
        output = (
            "Candidate answer" if model == battle_model else "score A: 1 score B: 0"
        )
        return ConstantModel(output)

    monkeypatch.setattr(utils, "make_model", fake_model)
    instructions = pd.Series(["Which answer is better?"], index=[1])
    completions = generate.generate_instructions(
        instructions,
        battle_model,
        use_tqdm=False,
        inference_cache=completion_cache,
    )
    metadata = [
        {
            "instruction_id": "1",
            "model_a": "candidate",
            "model_b": "baseline",
            "orientation": "direct",
        }
    ]
    arguments = {
        "instructions": instructions.tolist(),
        "completions_A": completions["completion"].tolist(),
        "completions_B": ["Baseline answer"],
        "cache_metadata": metadata,
    }

    first = evaluate.annotate_battles(
        judge_chat_model=prepare_model(judge_model, cache=judgement_cache),
        **arguments,
    )

    def fail_if_materialized(*_args, **_kwargs):
        raise AssertionError("cache hit materialized the model")

    monkeypatch.setattr(utils, "make_model", fail_if_materialized)
    cached_completions = generate.generate_instructions(
        instructions,
        battle_model,
        use_tqdm=False,
        inference_cache=completion_cache,
    )
    second = evaluate.annotate_battles(
        judge_chat_model=prepare_model(judge_model, cache=judgement_cache),
        **arguments,
    )

    assert cached_completions["completion"].tolist() == ["Candidate answer"]
    assert first[0].judge_completion == "score A: 1 score B: 0"
    assert second[0].judge_completion == first[0].judge_completion


def test_mixed_hits_and_misses_preserve_order(tmp_path, monkeypatch):
    cache = CompletionInferenceCache(tmp_path, "arena-hard")
    monkeypatch.setattr(
        utils, "make_model", lambda *_args, **_kwargs: ConstantModel("cached")
    )
    do_inference(
        prepare_model("Dummy/test-model", cache=cache),
        ["hit"],
        cache_metadata=[{"instruction_id": "hit"}],
    )

    backend = EchoModel()
    monkeypatch.setattr(utils, "make_model", lambda *_args, **_kwargs: backend)
    outputs = do_inference(
        prepare_model("Dummy/test-model", cache=cache),
        ["miss-a", "hit", "miss-b"],
        cache_metadata=[
            {"instruction_id": "a"},
            {"instruction_id": "hit"},
            {"instruction_id": "b"},
        ],
    )

    assert outputs == ["generated:miss-a", "cached", "generated:miss-b"]
    assert backend.calls == [["miss-a", "miss-b"]]


def test_multiturn_temperature_cache_full_hit(tmp_path, monkeypatch):
    questions = pd.DataFrame(
        {
            "turn_1": ["question 1", "question 2"],
            "turn_2": ["follow-up 1", "follow-up 2"],
            "category": ["writing", "math"],
        }
    )
    cache = CompletionInferenceCache(tmp_path, "mt-bench")
    kwargs = {"temperature_config": {"writing": 0.7, "math": 0.0}, "use_tqdm": False}
    first = generate.generate_multiturn(
        questions, "Dummy/answer", inference_cache=cache, **kwargs
    )

    def fail_if_materialized(*_args, **_kwargs):
        raise AssertionError("cache hit materialized a model")

    monkeypatch.setattr(utils, "make_model", fail_if_materialized)
    second = generate.generate_multiturn(
        questions, "Dummy/answer", inference_cache=cache, **kwargs
    )
    pd.testing.assert_frame_equal(second, first)


def test_vllm_descriptor_contains_output_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(inference.importlib_metadata, "version", lambda _name: "0.10.2")
    cache = CompletionInferenceCache(tmp_path, "arena-hard")

    model = prepare_model(
        "VLLM/Qwen/Qwen3-8B",
        max_tokens=32,
        cache=cache,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        tensor_parallel_size=2,
        temperature=0.2,
    )

    assert model.descriptor["backend_version"] == "0.10.2"
    assert model.descriptor["model_kwargs"] == {
        "max_tokens": 32,
        "max_model_len": 4096,
        "chat_template": None,
    }
    assert model.descriptor["sampling"] == {"temperature": 0.2, "top_p": 0.95}


@pytest.mark.parametrize(
    ("model_spec", "expected_mode", "expected_endpoint"),
    [
        ("Dummy/model", "chat", None),
        ("VLLM/org/model", "auto", None),
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
    unpinned = prepare_model(
        "OpenRouter/org/model",
        cache=cache,
    ).descriptor
    assert "uses unpinned provider routing" in caplog.text

    caplog.clear()
    pinned = prepare_model(
        "OpenRouter/org/model",
        cache=cache,
        extra_body={"provider": {"order": ["Together"], "allow_fallbacks": False}},
    ).descriptor
    assert "uses unpinned provider routing" not in caplog.text

    assert unpinned != pinned
    assert pinned["model_kwargs"]["extra_body"]["provider"]["order"] == ["Together"]

    gateway = prepare_model(
        "ChatOpenAI/model",
        cache=cache,
        base_url="https://gateway.example/v1/",
    ).descriptor
    assert gateway["endpoint"] == "https://gateway.example/v1"

    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.example/v1")
    ambient = prepare_model("ChatOpenAI/model", cache=cache).descriptor
    assert ambient["endpoint"] == "https://ambient.example/v1"


def test_unsupported_provider_runs_uncached(tmp_path, caplog):
    cache = CompletionInferenceCache(tmp_path, "arena-hard")

    model = prepare_model("Unsupported/model", cache=cache)

    assert model.descriptor is None
    assert "Caching is not supported" in caplog.text


@pytest.mark.parametrize(
    ("provider", "expected_type"),
    [("OpenRouter", "messages"), ("Together", "text"), ("VLLM", "auto")],
)
def test_input_canonicalization_matches_provider_mode(provider, expected_type):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "System"), ("user", "Question")]
    ).invoke({})

    payload = json.loads(
        canonicalize_model_input(prompt, provider_input_mode(provider))
    )

    assert payload["type"] == expected_type
    if expected_type in {"auto", "text"}:
        assert payload["text"] == prompt.to_string()
