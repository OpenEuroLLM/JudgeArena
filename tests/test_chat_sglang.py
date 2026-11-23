import sys
from types import SimpleNamespace

import pytest

import judgearena.models as models
from judgearena.cache.inference import build_model_descriptor


def _install_fake_sglang(monkeypatch, *, tokenizer_template="{{ messages }}"):
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured["engine_init"] = kwargs

        def generate(self, **kwargs):
            captured["generate_call"] = kwargs
            return [
                {
                    "text": "ok",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "output_top_logprobs": [[(-0.1, 7, "A"), (-2.3, 8, "B")]],
                    },
                }
                for _ in kwargs["prompt"]
            ]

    class FakeTokenizer:
        chat_template = tokenizer_template

        def apply_chat_template(self, messages, **kwargs):
            captured["template_call"] = {"messages": messages, "kwargs": kwargs}
            return "PROMPT:" + messages[-1]["content"]

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=FakeEngine))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: FakeTokenizer()
            )
        ),
    )
    return captured


def test_make_model_builds_sglang_engine_with_mapped_kwargs(monkeypatch):
    captured = _install_fake_sglang(monkeypatch)

    model = models.make_model(
        "SGLang/Qwen/Qwen2.5-0.5B-Instruct",
        max_tokens=64,
        max_model_len=4096,
        temperature=0.0,
        seed=3,
        tp_size=2,
        gpu_memory_utilization=0.9,
    )

    assert isinstance(model, models.ChatSGLang)
    assert captured["engine_init"] == {
        "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "context_length": 4096,
        "random_seed": 3,
        "tp_size": 2,
        "trust_remote_code": True,
    }
    assert model.sampling_params == {
        "max_new_tokens": 64,
        "temperature": 0.0,
        "top_p": models.VLLM_DEFAULT_TOP_P,
    }


def test_chat_sglang_applies_chat_template_and_returns_text(monkeypatch):
    captured = _install_fake_sglang(monkeypatch)
    model = models.ChatSGLang(model="m", max_tokens=8, disable_thinking=True)

    assert model.batch(["hello"]) == ["ok"]
    assert captured["generate_call"]["prompt"] == ["PROMPT:hello"]
    assert captured["template_call"]["kwargs"]["enable_thinking"] is False
    assert captured["template_call"]["kwargs"]["add_generation_prompt"] is True


def test_chat_sglang_falls_back_to_plain_text_without_template(monkeypatch):
    captured = _install_fake_sglang(monkeypatch, tokenizer_template=None)
    with pytest.warns(UserWarning, match="does not define a chat template"):
        model = models.ChatSGLang(model="m")

    model.batch(["raw prompt"])
    assert captured["generate_call"]["prompt"] == ["raw prompt"]
    assert "template_call" not in captured


def test_chat_sglang_returns_logprobs_and_usage(monkeypatch):
    captured = _install_fake_sglang(monkeypatch)
    model = models.ChatSGLang(model="m", top_logprobs=2)

    [result] = model.batch(["q"], usage_stage="judge")

    assert captured["generate_call"]["top_logprobs_num"] == 2
    assert captured["generate_call"]["return_logprob"] is True
    assert result.text == "ok"
    assert result.first_token_top_logprobs == {"A": -0.1, "B": -2.3}
    assert result.usage.input_tokens == 3
    assert result.usage.output_tokens == 2
    assert result.usage.total_tokens == 5


def test_sglang_cache_descriptor_records_backend_version(monkeypatch):
    monkeypatch.setattr(
        "judgearena.cache.inference.importlib_metadata.version",
        lambda name: f"{name}-1.0",
    )
    descriptor = build_model_descriptor("SGLang", "m", {"max_tokens": 8})

    assert descriptor["input_mode"] == "chat"
    assert descriptor["backend_version"] == "sglang-1.0"
    assert descriptor["model_kwargs"]["temperature"] == models.VLLM_DEFAULT_TEMPERATURE


def test_max_model_len_overrides_context_length_engine_option(monkeypatch):
    captured = _install_fake_sglang(monkeypatch)
    models.ChatSGLang(model="m", max_model_len=2048, context_length=8192)

    assert captured["engine_init"]["context_length"] == 2048


def test_sglang_judge_inherits_model_engine_kwargs():
    kwargs = models.build_default_judge_model_kwargs(
        "SGLang/Qwen/Qwen2.5-7B-Instruct",
        {"tp_size": 2, "mem_fraction_static": 0.7},
        judge_engine_kwargs_override={"tp_size": 1},
    )

    assert kwargs == {"tp_size": 1, "mem_fraction_static": 0.7}


def test_sglang_cache_descriptor_ignores_hardware_kwargs(monkeypatch):
    monkeypatch.setattr(
        "judgearena.cache.inference.importlib_metadata.version",
        lambda name: "1.0",
    )
    small = build_model_descriptor("SGLang", "m", {"tp_size": 1})
    large = build_model_descriptor(
        "SGLang", "m", {"tp_size": 4, "dp_size": 2, "mem_fraction_static": 0.9}
    )

    assert small == large
