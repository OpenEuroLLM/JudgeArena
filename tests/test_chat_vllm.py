import sys
from types import SimpleNamespace

import pytest

import judgearena.utils as utils


def _install_fake_vllm(monkeypatch):
    captured = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["sampling_kwargs"] = kwargs

    class FakeReasoningConfig:
        def __init__(self, **kwargs):
            captured["reasoning_config_kwargs"] = kwargs

    class FakeLLM:
        def __init__(self, *, model, trust_remote_code, **kwargs):
            captured["llm_init"] = {
                "model": model,
                "trust_remote_code": trust_remote_code,
                "kwargs": kwargs,
            }

        def get_tokenizer(self):
            return SimpleNamespace(chat_template="{{ messages }}")

        def chat(self, messages, sampling_params, **kwargs):
            captured["chat_call"] = {
                "messages": messages,
                "sampling_params": sampling_params,
                "kwargs": kwargs,
            }
            return [SimpleNamespace(outputs=[SimpleNamespace(text="ok")])]

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
    return captured, FakeReasoningConfig


def test_chat_vllm_enables_reasoning_support_for_qwen_thinking_budget(monkeypatch):
    captured, fake_reasoning_config = _install_fake_vllm(monkeypatch)

    utils.ChatVLLM(
        model="Qwen/Qwen3.5-27B-FP8",
        max_tokens=128,
        thinking_token_budget=64,
        gpu_memory_utilization=0.7,
    )

    assert captured["sampling_kwargs"]["thinking_token_budget"] == 64
    assert "structured_outputs" not in captured["sampling_kwargs"]
    assert captured["reasoning_config_kwargs"] == {
        "reasoning_start_str": utils.VLLM_QWEN_REASONING_START_STR,
        "reasoning_end_str": utils.VLLM_QWEN_REASONING_END_STR,
    }
    llm_kwargs = captured["llm_init"]["kwargs"]
    assert llm_kwargs["reasoning_parser"] == "qwen3"
    assert isinstance(llm_kwargs["reasoning_config"], fake_reasoning_config)


def test_chat_vllm_clamps_thinking_budget_to_total_max_tokens(monkeypatch):
    captured, _fake_reasoning_config = _install_fake_vllm(monkeypatch)

    utils.ChatVLLM(
        model="Qwen/Qwen3.5-27B-FP8",
        max_tokens=32,
        thinking_token_budget=64,
        gpu_memory_utilization=0.7,
    )

    assert captured["sampling_kwargs"]["thinking_token_budget"] == 32


def test_chat_vllm_passes_disable_thinking_via_chat_template_kwargs(monkeypatch):
    captured, _fake_reasoning_config = _install_fake_vllm(monkeypatch)
    chat_model = utils.ChatVLLM(
        model="Qwen/Qwen3.5-27B-FP8",
        max_tokens=16,
        disable_thinking=True,
        gpu_memory_utilization=0.7,
    )

    outputs = chat_model.batch(["hello"])

    assert outputs == ["ok"]
    assert captured["chat_call"]["kwargs"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_build_default_judge_model_kwargs_only_defaults_qwen_judges():
    assert utils.build_default_judge_model_kwargs(
        "VLLM/Qwen/Qwen3.5-27B-FP8",
        {"gpu_memory_utilization": 0.7},
    ) == {
        "gpu_memory_utilization": 0.7,
        "thinking_token_budget": 512,
    }
    assert utils.build_default_judge_model_kwargs(
        "VLLM/meta-llama/Llama-3.3-70B-Instruct",
        {"gpu_memory_utilization": 0.7},
    ) == {"gpu_memory_utilization": 0.7}
    assert (
        utils.build_default_judge_model_kwargs(
            "OpenRouter/qwen/qwen3-32b",
            {},
        )
        == {}
    )


def test_chat_vllm_preserves_explicit_reasoning_settings_for_non_qwen(monkeypatch):
    captured, _fake_reasoning_config = _install_fake_vllm(monkeypatch)
    explicit_reasoning_config = object()

    utils.ChatVLLM(
        model="meta-llama/Llama-3.3-70B-Instruct",
        max_tokens=16,
        thinking_token_budget=32,
        reasoning_parser="custom-parser",
        reasoning_config=explicit_reasoning_config,
        gpu_memory_utilization=0.7,
    )

    assert captured["sampling_kwargs"]["thinking_token_budget"] == 16
    assert captured["llm_init"]["kwargs"]["reasoning_parser"] == "custom-parser"
    assert (
        captured["llm_init"]["kwargs"]["reasoning_config"] is explicit_reasoning_config
    )


def test_chat_vllm_ignores_thinking_budget_for_unknown_family(monkeypatch):
    captured, _fake_reasoning_config = _install_fake_vllm(monkeypatch)

    with pytest.warns(UserWarning, match="built-in Qwen reasoning defaults"):
        utils.ChatVLLM(
            model="meta-llama/Llama-3.3-70B-Instruct",
            max_tokens=32,
            thinking_token_budget=64,
            gpu_memory_utilization=0.7,
        )

    assert "thinking_token_budget" not in captured["sampling_kwargs"]
    assert "reasoning_parser" not in captured["llm_init"]["kwargs"]
    assert "reasoning_config" not in captured["llm_init"]["kwargs"]


def test_infer_model_spec_uses_type_based_vllm_fallback():
    model = object.__new__(utils.ChatVLLM)
    model.model_path = "Qwen/Qwen3.5-27B-FP8"

    assert utils.infer_model_spec_from_instance(model) == "VLLM/Qwen/Qwen3.5-27B-FP8"


def test_infer_model_spec_uses_type_based_llamacpp_fallback(monkeypatch):
    class FakeLlamaCpp:
        def __init__(self, model_path: str):
            self.model_path = model_path

    monkeypatch.setattr(utils, "LlamaCpp", FakeLlamaCpp)
    model = FakeLlamaCpp("./models/model.gguf")

    assert utils.infer_model_spec_from_instance(model) == "LlamaCpp/./models/model.gguf"
