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
        model="Qwen/Qwen3.5-9B",
        max_tokens=128,
        thinking_token_budget=64,
        gpu_memory_utilization=0.7,
    )

    assert captured["sampling_kwargs"]["thinking_token_budget"] == 64
    assert "structured_outputs" not in captured["sampling_kwargs"]
    assert captured["reasoning_config_kwargs"] == {
        "reasoning_start_str": utils.VLLM_REASONING_START_STR,
        "reasoning_end_str": utils.VLLM_REASONING_END_STR,
    }
    llm_kwargs = captured["llm_init"]["kwargs"]
    assert llm_kwargs["reasoning_parser"] == "qwen3"
    assert isinstance(llm_kwargs["reasoning_config"], fake_reasoning_config)


def test_chat_vllm_enables_reasoning_support_for_smollm3_thinking_budget(monkeypatch):
    captured, fake_reasoning_config = _install_fake_vllm(monkeypatch)

    utils.ChatVLLM(
        model="HuggingFaceTB/SmolLM3-3B",
        max_tokens=128,
        thinking_token_budget=64,
        gpu_memory_utilization=0.7,
    )

    assert captured["sampling_kwargs"]["thinking_token_budget"] == 64
    assert captured["reasoning_config_kwargs"] == {
        "reasoning_start_str": utils.VLLM_REASONING_START_STR,
        "reasoning_end_str": utils.VLLM_REASONING_END_STR,
    }
    llm_kwargs = captured["llm_init"]["kwargs"]
    assert llm_kwargs["reasoning_parser"] == "qwen3"
    assert isinstance(llm_kwargs["reasoning_config"], fake_reasoning_config)


def test_chat_vllm_clamps_thinking_budget_to_total_max_tokens(monkeypatch):
    captured, _fake_reasoning_config = _install_fake_vllm(monkeypatch)

    utils.ChatVLLM(
        model="Qwen/Qwen3.5-9B",
        max_tokens=32,
        thinking_token_budget=64,
        gpu_memory_utilization=0.7,
    )

    assert captured["sampling_kwargs"]["thinking_token_budget"] == 32


def test_chat_vllm_passes_disable_thinking_via_chat_template_kwargs(monkeypatch):
    captured, _fake_reasoning_config = _install_fake_vllm(monkeypatch)
    chat_model = utils.ChatVLLM(
        model="Qwen/Qwen3.5-9B",
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
        "VLLM/Qwen/Qwen3.5-9B",
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


def test_build_default_judge_model_kwargs_sets_fp8_kv_cache_for_fp8_judges():
    fp8_defaults = utils.build_default_judge_model_kwargs(
        "VLLM/Skywork/Skywork-Critic-Llama-3.1-70B-FP8",
        {"gpu_memory_utilization": 0.9},
    )
    assert fp8_defaults["kv_cache_dtype"] == "fp8"
    # FP8 Skywork judge is not Qwen3/SmolLM3 so no thinking-token default.
    assert "thinking_token_budget" not in fp8_defaults

    bf16_defaults = utils.build_default_judge_model_kwargs(
        "VLLM/Skywork/Skywork-Critic-Llama-3.1-8B",
        {"gpu_memory_utilization": 0.9},
    )
    assert "kv_cache_dtype" not in bf16_defaults

    explicit_override = utils.build_default_judge_model_kwargs(
        "VLLM/Skywork/Skywork-Critic-Llama-3.1-70B-FP8",
        {"gpu_memory_utilization": 0.9, "kv_cache_dtype": "bfloat16"},
    )
    assert explicit_override["kv_cache_dtype"] == "bfloat16"

    # Non-VLLM providers never receive the FP8 KV default even if the name
    # happens to contain "fp8".
    non_vllm = utils.build_default_judge_model_kwargs("OpenRouter/some/Model-fp8", {})
    assert "kv_cache_dtype" not in non_vllm


def test_build_default_judge_model_kwargs_overlays_judge_override():
    """Judge-scoped overrides must win over shared ``engine_kwargs`` so the
    battle engine can stay on TP=1 while the 70B FP8 judge pins TP=2."""
    merged = utils.build_default_judge_model_kwargs(
        "VLLM/Skywork/Skywork-Critic-Llama-3.1-70B-FP8",
        {"gpu_memory_utilization": 0.9},
        judge_engine_kwargs_override={"tensor_parallel_size": 2},
    )
    assert merged["tensor_parallel_size"] == 2
    assert merged["gpu_memory_utilization"] == 0.9
    assert merged["kv_cache_dtype"] == "fp8"

    overridden = utils.build_default_judge_model_kwargs(
        "VLLM/Skywork/Skywork-Critic-Llama-3.1-70B-FP8",
        {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9},
        judge_engine_kwargs_override={"tensor_parallel_size": 4},
    )
    assert overridden["tensor_parallel_size"] == 4
    # FP8 weights + FP8 KV cache are a name-driven invariant; the TP override
    # must not silently drop `kv_cache_dtype=fp8` because we run the Skywork
    # 70B FP8 judge on TP=2 and TP=4 interchangeably depending on the cell.
    assert overridden["kv_cache_dtype"] == "fp8"

    empty_override = utils.build_default_judge_model_kwargs(
        "VLLM/Skywork/Skywork-Critic-Llama-3.1-70B-FP8",
        {"tensor_parallel_size": 1},
        judge_engine_kwargs_override={},
    )
    assert empty_override["tensor_parallel_size"] == 1


def test_is_thinking_model_matches_qwen3_and_smollm3_repo_ids():
    assert utils.is_thinking_model("Qwen/Qwen3.5-9B")
    assert utils.is_thinking_model("HuggingFaceTB/SmolLM3-3B")
    assert utils.is_thinking_model("Qwen/Qwen3-7B")
    assert not utils.is_thinking_model("Qwen/Qwen2.5-7B")
    assert not utils.is_thinking_model("utter-project/EuroLLM-9B-Instruct")
    assert not utils.is_thinking_model("meta-llama/Llama-3.1-8B")


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

    with pytest.warns(UserWarning, match="built-in thinking-model"):
        utils.ChatVLLM(
            model="meta-llama/Llama-3.3-70B-Instruct",
            max_tokens=32,
            thinking_token_budget=64,
            gpu_memory_utilization=0.7,
        )

    assert "thinking_token_budget" not in captured["sampling_kwargs"]
    assert "reasoning_parser" not in captured["llm_init"]["kwargs"]
    assert "reasoning_config" not in captured["llm_init"]["kwargs"]


def test_chat_vllm_records_thinking_budget_exhaustion_metadata(monkeypatch):
    captured, _fake_reasoning_config = _install_fake_vllm(monkeypatch)

    class FakeLLMWithMarker:
        def __init__(self, *, model, trust_remote_code, **kwargs):
            captured["llm_init"] = {"model": model, "kwargs": kwargs}

        def get_tokenizer(self):
            return SimpleNamespace(chat_template="{{ messages }}")

        def chat(self, messages, sampling_params, **kwargs):
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text=f"pre {utils.VLLM_REASONING_END_STR} answer",
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ]
                ),
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="clean answer",
                            finish_reason="stop",
                            stop_reason=None,
                        )
                    ]
                ),
            ]

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=FakeLLMWithMarker,
            SamplingParams=sys.modules["vllm"].SamplingParams,
        ),
    )

    chat_model = utils.ChatVLLM(
        model="Qwen/Qwen3.5-9B",
        max_tokens=64,
        thinking_token_budget=32,
        gpu_memory_utilization=0.7,
    )
    _texts, metadata = chat_model.batch_with_metadata(["a", "b"])

    assert metadata[0]["thinking_budget_exhausted"] is True
    assert metadata[0]["thinking_token_budget"] == 32
    assert metadata[1]["thinking_budget_exhausted"] is False
    assert metadata[1]["thinking_token_budget"] == 32


def test_chat_vllm_omits_thinking_budget_metadata_without_budget(monkeypatch):
    _captured, _fake_reasoning_config = _install_fake_vllm(monkeypatch)

    chat_model = utils.ChatVLLM(
        model="Qwen/Qwen3.5-9B",
        max_tokens=64,
        gpu_memory_utilization=0.7,
    )
    assert chat_model._thinking_budget_marker is None
    assert chat_model._thinking_budget_value is None


def test_infer_model_spec_uses_type_based_vllm_fallback():
    model = object.__new__(utils.ChatVLLM)
    model.model_path = "Qwen/Qwen3.5-9B"

    assert utils.infer_model_spec_from_instance(model) == "VLLM/Qwen/Qwen3.5-9B"


def test_infer_model_spec_uses_type_based_llamacpp_fallback(monkeypatch):
    class FakeLlamaCpp:
        def __init__(self, model_path: str):
            self.model_path = model_path

    monkeypatch.setattr(utils, "LlamaCpp", FakeLlamaCpp)
    model = FakeLlamaCpp("./models/model.gguf")

    assert utils.infer_model_spec_from_instance(model) == "LlamaCpp/./models/model.gguf"
