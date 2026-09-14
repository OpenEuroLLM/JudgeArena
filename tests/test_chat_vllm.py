import json
import sys
from types import SimpleNamespace

import judgearena.models as models


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
            captured["llm_kwargs"] = kwargs

        def get_tokenizer(self):
            return SimpleNamespace(chat_template="{{ messages }}")

        def chat(self, messages, sampling_params, **kwargs):
            captured["chat_kwargs"] = kwargs
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

    models.ChatVLLM(
        model="Qwen/Qwen3.5-9B",
        max_tokens=128,
        thinking_token_budget=64,
        gpu_memory_utilization=0.7,
    )

    assert captured["sampling_kwargs"]["thinking_token_budget"] == 64
    assert "structured_outputs" not in captured["sampling_kwargs"]
    assert captured["reasoning_config_kwargs"] == {
        "reasoning_start_str": models.VLLM_REASONING_START_STR,
        "reasoning_end_str": models.VLLM_REASONING_END_STR,
    }
    llm_kwargs = captured["llm_kwargs"]
    assert llm_kwargs["reasoning_parser"] == "qwen3"
    assert isinstance(llm_kwargs["reasoning_config"], fake_reasoning_config)


def test_chat_vllm_uses_olmo3_reasoning_parser_for_olmo_think(monkeypatch):
    captured, _ = _install_fake_vllm(monkeypatch)
    models.ChatVLLM(model="allenai/Olmo-3-7B-Think", thinking_token_budget=64)
    assert captured["llm_kwargs"]["reasoning_parser"] == "olmo3"


def test_chat_vllm_passes_disable_thinking_via_chat_template_kwargs(monkeypatch):
    captured, _fake_reasoning_config = _install_fake_vllm(monkeypatch)
    chat_model = models.ChatVLLM(
        model="Qwen/Qwen3.5-9B",
        max_tokens=16,
        disable_thinking=True,
        gpu_memory_utilization=0.7,
    )

    outputs = chat_model.batch(["hello"])

    assert outputs == ["ok"]
    assert captured["chat_kwargs"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_build_default_judge_model_kwargs_scopes_thinking_defaults_to_vllm():
    defaults = models.build_default_judge_model_kwargs(
        "VLLM/Qwen/Qwen3.5-9B", {"gpu_memory_utilization": 0.7}
    )
    assert defaults["thinking_token_budget"] == 512
    assert (
        models.build_default_judge_model_kwargs(
            "OpenRouter/qwen/qwen3-32b", {"gpu_memory_utilization": 0.7}
        )
        == {}
    )


def test_build_default_judge_model_kwargs_respects_explicit_kv_cache():
    defaults = models.build_default_judge_model_kwargs(
        "VLLM/Skywork/Skywork-Critic-Llama-3.1-70B-FP8", {"kv_cache_dtype": "bfloat16"}
    )
    assert defaults["kv_cache_dtype"] == "bfloat16"


def test_build_default_judge_model_kwargs_overlays_judge_override():
    overridden = models.build_default_judge_model_kwargs(
        "VLLM/Skywork/Skywork-Critic-Llama-3.1-70B-FP8",
        {"tensor_parallel_size": 1},
        judge_engine_kwargs_override={"tensor_parallel_size": 4},
    )
    assert overridden["tensor_parallel_size"] == 4
    assert overridden["kv_cache_dtype"] == "fp8"


def test_is_thinking_model_matches_smollm3_but_not_older_qwen():
    assert models.is_thinking_model("HuggingFaceTB/SmolLM3-3B")
    assert not models.is_thinking_model("Qwen/Qwen2.5-7B")


def test_chat_vllm_preserves_explicit_reasoning_settings_for_non_qwen(monkeypatch):
    captured, _fake_reasoning_config = _install_fake_vllm(monkeypatch)
    explicit_reasoning_config = object()

    models.ChatVLLM(
        model="meta-llama/Llama-3.3-70B-Instruct",
        max_tokens=16,
        thinking_token_budget=32,
        reasoning_parser="custom-parser",
        reasoning_config=explicit_reasoning_config,
        gpu_memory_utilization=0.7,
    )

    assert captured["sampling_kwargs"]["thinking_token_budget"] == 16
    assert captured["llm_kwargs"]["reasoning_parser"] == "custom-parser"
    assert captured["llm_kwargs"]["reasoning_config"] is explicit_reasoning_config


def test_is_retryable_error_retries_jsondecodeerror():
    err = json.JSONDecodeError("Expecting value", "<html>503</html>", 0)
    assert models._is_retryable_error(err) is True


def test_is_retryable_error_retries_transient_http_codes():
    assert models._is_retryable_error(Exception("HTTP 503 Service Unavailable")) is True


def test_is_retryable_error_does_not_retry_auth_failure():
    assert models._is_retryable_error(Exception("401 User not found.")) is False
