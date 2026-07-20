import json
import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompt_values import ChatPromptValue

import judgearena.model_adapters as adapters
import judgearena.models as models
from judgearena.model_adapters import (
    HOSTED_ADAPTER_VERSION,
    LOCAL_LLAMACPP_ADAPTER_VERSION,
    build_producer_metadata,
    build_vllm_descriptor,
    effective_sampling,
    normalize_constructor_settings,
    top_k_from_settings,
    wrap_known_model,
)
from judgearena.models import DummyModel, make_model, resolve_vllm_settings


def test_normalize_constructor_settings_redacts_secret_keys():
    settings = {
        "temperature": 0.5,
        "api_key": "secret-value",
        "default_headers": {"Authorization": "Bearer x"},
        "model_kwargs": {"top_k": 40},
    }
    normalized = normalize_constructor_settings(settings)
    assert normalized == {"temperature": 0.5, "model_kwargs": {"top_k": 40}}
    serialized = json.dumps(normalized)
    assert "secret-value" not in serialized
    assert "Authorization" not in serialized


def test_make_model_openrouter_endpoint_unifies_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    model = make_model(
        "ChatOpenAI/google/gemma-3-4b-it",
        max_tokens=16,
        base_url="https://openrouter.ai/api/v1",
    )
    descriptor = model.cache_descriptor()
    assert descriptor is not None
    assert model.provider == "OpenRouter"
    assert model.model_spec == "OpenRouter/google/gemma-3-4b-it"
    assert descriptor["provider"] == "OpenRouter"
    assert descriptor["base_url"] == "https://openrouter.ai/api/v1"


def test_lazy_and_materialized_hosted_descriptors_match(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("TOGETHER_API_KEY", "dummy")
    specs = [
        "OpenRouter/openai/gpt-4o-mini",
        "OpenAI/gpt-3.5-turbo-instruct",
        "Together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ]

    for spec in specs:
        lazy = make_model(spec, max_tokens=8, temperature=0.2)
        wrapped = wrap_known_model(lazy.materialize(), model_spec=lazy.model_spec)
        assert wrapped is not None
        assert wrapped.cache_descriptor() == lazy.cache_descriptor()


def test_wrapped_hosted_model_preserves_unset_max_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    backend = models.ChatOpenAI(model="gpt-4o-mini")

    wrapped = wrap_known_model(backend)

    assert wrapped is not None
    assert wrapped.max_tokens is None
    assert "max_tokens" not in wrapped.cache_descriptor()["sampling"]


def test_make_model_captures_base_url_for_generic_hosted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    model = make_model(
        "OpenAI/text-davinci-003",
        max_tokens=8,
        openai_api_base="https://example.test/v1",
    )
    descriptor = model.cache_descriptor()
    assert descriptor is not None
    assert descriptor["base_url"] == "https://example.test/v1"


def test_make_model_preserves_constructor_secrets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    model = make_model(
        "OpenRouter/google/gemma-3-4b-it",
        max_tokens=16,
        api_key="runtime-secret",
        default_headers={"Authorization": "Bearer secret"},
    )
    assert model.engine_kwargs["api_key"] == "runtime-secret"
    assert model.engine_kwargs["default_headers"]["Authorization"] == "Bearer secret"
    descriptor = model.cache_descriptor()
    assert descriptor is not None
    serialized = json.dumps(descriptor)
    assert "runtime-secret" not in serialized
    assert "Authorization" not in serialized


def test_top_k_from_settings_preserves_zero():
    assert top_k_from_settings({"top_k": 0}) == 0
    assert top_k_from_settings({"model_kwargs": {"top_k": 0}}) == 0
    assert (
        effective_sampling(
            temperature=0.5,
            top_p=0.9,
            top_k=top_k_from_settings({"top_k": 0}),
            seed=None,
        )["top_k"]
        == 0
    )


def test_resolve_vllm_settings_uses_explicit_tokenizer(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.config.reasoning",
        SimpleNamespace(
            ReasoningConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )

    seen: dict[str, object] = {}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model, **kwargs):
            seen["tokenizer_id"] = model
            seen["tokenizer_kwargs"] = kwargs
            return SimpleNamespace(chat_template="{{ messages }}")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model, **kwargs):
            return SimpleNamespace(max_position_embeddings=4096)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeAutoTokenizer,
            AutoConfig=FakeAutoConfig,
        ),
    )

    resolve_vllm_settings(
        "org/base-model",
        max_tokens=16,
        tokenizer="org/custom-tokenizer",
        revision="model-rev",
        tokenizer_revision="tok-rev",
        trust_remote_code=False,
    )
    assert seen["tokenizer_id"] == "org/custom-tokenizer"
    assert seen["tokenizer_kwargs"] == {
        "trust_remote_code": False,
        "revision": "tok-rev",
    }


def test_vllm_descriptor_uses_fully_resolved_sampling(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.config.reasoning",
        SimpleNamespace(
            ReasoningConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model, **kwargs):
            return SimpleNamespace(chat_template="{{ messages }}")

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model, **kwargs):
            return SimpleNamespace(max_position_embeddings=4096)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=FakeAutoTokenizer,
            AutoConfig=FakeAutoConfig,
        ),
    )

    resolved = resolve_vllm_settings(
        "Qwen/Qwen3.5-9B",
        max_tokens=32,
        max_model_len=8192,
        thinking_token_budget=16,
        gpu_memory_utilization=0.7,
    )
    descriptor = build_vllm_descriptor("VLLM/Qwen/Qwen3.5-9B", resolved)
    assert descriptor is not None
    assert descriptor["sampling"]["temperature"] == 0.6
    assert descriptor["sampling"]["top_p"] == 0.95
    assert descriptor["sampling"]["max_tokens"] == 32
    assert descriptor["sampling"]["thinking_token_budget"] == 16
    assert descriptor["engine_settings"]["max_model_len"] == 4096
    assert descriptor["engine_settings"]["gpu_memory_utilization"] == 0.7
    assert descriptor["engine_settings"]["reasoning_parser"] == "qwen3"
    assert "reasoning_config" in descriptor["engine_settings"]
    assert "vllm_version" in descriptor
    assert "langchain_openai_version" not in descriptor


def test_vllm_set_temperature_updates_sampling_only():
    descriptor = {
        "sampling": {"temperature": 0.6, "max_tokens": 8},
        "engine_settings": {"max_model_len": 1024},
    }
    resolved = SimpleNamespace(sampling_params_kwargs={"temperature": 0.6})
    model = adapters.PreparedModel(
        provider="VLLM",
        model_spec="VLLM/test/model",
        model_name="test/model",
        max_tokens=8,
        engine_kwargs={"max_model_len": 1024},
        sampling={"temperature": 0.6, "max_tokens": 8},
        input_mode="chat",
        descriptor=descriptor,
        materialize=lambda _prepared: (_ for _ in ()).throw(
            AssertionError("must stay lazy")
        ),
        producer_metadata={},
        vllm_resolved=resolved,
    )

    model.set_temperature(0.9)

    updated = model.cache_descriptor()
    assert updated["sampling"]["temperature"] == 0.9
    assert "temperature" not in updated["engine_settings"]
    assert resolved.sampling_params_kwargs["temperature"] == 0.9


def test_llamacpp_descriptor_includes_local_engine_version(monkeypatch):
    monkeypatch.setattr(
        adapters,
        "_provider_package_version",
        lambda name: "0.42.0" if name == "llama-cpp-python" else "1.0.0",
    )
    model = make_model("LlamaCpp/models/test.gguf", max_tokens=64)
    descriptor = model.cache_descriptor()
    assert descriptor is not None
    assert descriptor["llama_cpp_python_version"] == "0.42.0"
    assert descriptor["local_adapter_version"] == LOCAL_LLAMACPP_ADAPTER_VERSION
    metadata = model.producer_metadata()
    assert metadata["llama_cpp_python_version"] == "0.42.0"
    assert metadata["descriptor_schema_version"] == adapters.DESCRIPTOR_SCHEMA_VERSION


def test_hosted_canonicalization_preserves_tool_fields():
    prompt = ChatPromptValue(
        messages=[
            SystemMessage(content="sys"),
            HumanMessage(content="call tool"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "lookup",
                        "args": {"q": "paris"},
                    }
                ],
            ),
            ToolMessage(content='{"city":"Paris"}', tool_call_id="call-1"),
        ]
    )
    canonical = json.loads(adapters.canonicalize_hosted_chat_input(prompt))
    assert canonical["messages"][0] == {"role": "system", "content": "sys"}
    tool_message = canonical["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call-1"
    assert tool_message["content"] == '{"city":"Paris"}'
    ai_message = canonical["messages"][2]
    assert ai_message["tool_calls"][0]["name"] == "lookup"
    assert ai_message["tool_calls"][0]["id"] == "call-1"
    assert "id" not in ai_message


def test_vllm_dict_message_roles_match_runtime_normalization():
    messages = [
        {"role": "human", "content": "hello"},
        {"content": "missing role"},
    ]

    normalized = adapters.vllm_input_to_messages(messages)
    canonical = json.loads(
        adapters.canonicalize_vllm_input(messages, input_mode="chat")
    )

    assert normalized == [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "missing role"},
    ]
    assert canonical["messages"] == normalized


def test_raw_canonicalization_accepts_dict_messages():
    canonical = json.loads(
        adapters.canonicalize_raw_input(
            [{"role": "user", "content": "hello"}, {"content": "world"}]
        )
    )
    assert canonical == {"kind": "raw", "text": "hello\nworld"}


def test_set_temperature_updates_materialized_chatopenai_temperature(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    class FakeChat:
        model_fields = {
            "temperature": object(),
            "max_tokens": object(),
            "model": object(),
        }

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(models, "ChatOpenAI", FakeChat)
    model = make_model("OpenRouter/openai/gpt-4o-mini", max_tokens=8, temperature=0.2)
    backend = model.materialize()
    model.set_temperature(0.8)
    assert backend.temperature == 0.8
    assert model.cache_descriptor()["sampling"]["temperature"] == 0.8


def test_wrap_known_dummy_model():
    backend = DummyModel("Dummy/wrapped", max_tokens=8, temperature=0.1)
    wrapped = wrap_known_model(backend)
    assert wrapped is not None
    assert wrapped.cache_descriptor() is not None
    assert wrapped.canonicalize_input("hello").startswith('{"kind"')


def test_producer_metadata_includes_adapter_schema():
    metadata = build_producer_metadata(provider="OpenRouter")
    assert metadata["hosted_adapter_version"] == HOSTED_ADAPTER_VERSION
    assert metadata["descriptor_schema_version"] == adapters.DESCRIPTOR_SCHEMA_VERSION
    assert "langchain_openai_version" in metadata
