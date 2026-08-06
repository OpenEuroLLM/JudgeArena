import json
import sys
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompt_values import ChatPromptValue

import judgearena.model_adapters as adapters
from judgearena.model_adapters import (
    HOSTED_ADAPTER_VERSION,
    build_producer_metadata,
    build_vllm_descriptor,
    normalize_constructor_settings,
    wrap_known_model,
)
from judgearena.models import make_model, resolve_vllm_settings


def test_normalize_constructor_settings_redacts_secret_keys():
    normalized = normalize_constructor_settings(
        {
            "temperature": 0.5,
            "api_key": "secret-value",
            "default_headers": {"Authorization": "Bearer x"},
            "model_kwargs": {"top_k": 40},
        }
    )

    assert normalized == {"temperature": 0.5, "model_kwargs": {"top_k": 40}}
    serialized = json.dumps(normalized)
    assert "secret-value" not in serialized
    assert "Authorization" not in serialized


def test_openrouter_endpoint_unifies_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    model = make_model(
        "ChatOpenAI/google/gemma-3-4b-it",
        max_tokens=16,
        base_url="https://openrouter.ai/api/v1",
    )

    descriptor = model.cache_descriptor()
    assert model.model_spec == "OpenRouter/google/gemma-3-4b-it"
    assert descriptor["provider"] == "OpenRouter"
    assert descriptor["base_url"] == "https://openrouter.ai/api/v1"


def test_lazy_and_materialized_hosted_descriptors_match(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("TOGETHER_API_KEY", "dummy")

    for spec in (
        "OpenRouter/openai/gpt-4o-mini",
        "OpenAI/gpt-3.5-turbo-instruct",
        "Together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    ):
        lazy = make_model(spec, max_tokens=8, temperature=0.2)
        wrapped = wrap_known_model(lazy.materialize(), model_spec=lazy.model_spec)
        assert wrapped is not None
        assert wrapped.cache_descriptor() == lazy.cache_descriptor()


def test_vllm_descriptor_uses_fully_resolved_sampling(monkeypatch):
    monkeypatch.setattr(
        adapters,
        "_provider_package_version",
        lambda name: "test-version",
    )
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

    assert descriptor["sampling"] == {
        "max_tokens": 32,
        "temperature": 0.6,
        "thinking_token_budget": 16,
        "top_p": 0.95,
    }
    assert descriptor["engine_settings"]["max_model_len"] == 4096
    assert descriptor["engine_settings"]["reasoning_parser"] == "qwen3"
    assert "reasoning_config" in descriptor["engine_settings"]
    assert "vllm_version" in descriptor


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

    messages = json.loads(adapters.canonicalize_hosted_chat_input(prompt))["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[2]["tool_calls"][0]["name"] == "lookup"
    assert messages[3] == {
        "role": "tool",
        "content": '{"city":"Paris"}',
        "tool_call_id": "call-1",
    }


def test_vllm_canonicalization_matches_runtime_role_normalization():
    messages = [
        {"role": "human", "content": "hello"},
        {"content": "missing role"},
    ]
    normalized = adapters.vllm_input_to_messages(messages)
    canonical = json.loads(
        adapters.canonicalize_vllm_input(messages, input_mode="chat")
    )

    assert (
        canonical["messages"]
        == normalized
        == [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "missing role"},
        ]
    )


def test_producer_metadata_includes_adapter_schema():
    metadata = build_producer_metadata(provider="OpenRouter")
    assert metadata["hosted_adapter_version"] == HOSTED_ADAPTER_VERSION
    assert metadata["descriptor_schema_version"] == adapters.DESCRIPTOR_SCHEMA_VERSION
