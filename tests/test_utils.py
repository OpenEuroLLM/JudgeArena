from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import judgearena.utils as utils
from judgearena.utils import make_model


def test_extract_ai_message_metadata_reads_finish_reason():
    ai_message = SimpleNamespace(
        content="hi",
        response_metadata={"finish_reason": "length", "stop_reason": None},
    )
    md = utils._extract_ai_message_metadata(ai_message)
    assert md == {"finish_reason": "length", "stop_reason": None}


def test_extract_ai_message_metadata_handles_missing_response_metadata():
    bare_ai_message = SimpleNamespace(content="hello")
    md = utils._extract_ai_message_metadata(bare_ai_message)
    assert md == {"finish_reason": None, "stop_reason": None}


def test_extract_ai_message_metadata_handles_plain_dict_fallback():
    md = utils._extract_ai_message_metadata(
        {"finish_reason": "stop", "stop_reason": "eos"}
    )
    assert md == {"finish_reason": "stop", "stop_reason": "eos"}


def test_do_inference_async_path_propagates_finish_reason(monkeypatch):
    async_results = [
        SimpleNamespace(
            content="out1",
            response_metadata={"finish_reason": "stop"},
        ),
        SimpleNamespace(
            content="out2",
            response_metadata={"finish_reason": "length"},
        ),
    ]

    async def fake_ainvoke(_input, **_kwargs):
        return async_results.pop(0)

    chat_model = SimpleNamespace(ainvoke=fake_ainvoke)
    texts, metadata = utils.do_inference(
        chat_model=chat_model,
        inputs=["prompt1", "prompt2"],
        use_tqdm=True,
        return_metadata=True,
    )
    assert texts == ["out1", "out2"]
    assert metadata == [
        {"finish_reason": "stop", "stop_reason": None},
        {"finish_reason": "length", "stop_reason": None},
    ]


def test_do_inference_async_path_uses_single_item_batch_metadata():
    class PlainTextAsyncModel:
        async def ainvoke(self, input_item, **_kwargs):
            return f"plain-{input_item}"

        def batch_with_metadata(self, inputs, **_kwargs):
            assert inputs == ["prompt"]
            return ["out"], [{"finish_reason": "length", "stop_reason": None}]

    texts, metadata = utils.do_inference(
        chat_model=PlainTextAsyncModel(),
        inputs=["prompt"],
        use_tqdm=True,
        return_metadata=True,
    )

    assert texts == ["out"]
    assert metadata == [{"finish_reason": "length", "stop_reason": None}]


def test_do_inference_batch_path_propagates_finish_reason_without_batch_with_metadata():
    batch_results = [
        SimpleNamespace(
            content="a",
            response_metadata={"finish_reason": "stop"},
        ),
        SimpleNamespace(
            content="b",
            response_metadata={"finish_reason": "length"},
        ),
    ]
    chat_model = MagicMock()
    chat_model.batch = MagicMock(return_value=batch_results)
    # Ensure no batch_with_metadata attr so the else branch runs
    if hasattr(chat_model, "batch_with_metadata"):
        del chat_model.batch_with_metadata

    texts, metadata = utils.do_inference(
        chat_model=chat_model,
        inputs=["p1", "p2"],
        use_tqdm=False,
        return_metadata=True,
    )
    assert [m["finish_reason"] for m in metadata] == ["stop", "length"]
    assert texts == ["a", "b"]


def _strip_then_cap(
    answer: str, cap: int, *, strip: bool = True
) -> tuple[str, bool, bool]:
    if strip:
        stripped_text, thinking_stripped = utils.strip_thinking_tags_with_metadata(
            answer
        )
    else:
        stripped_text, thinking_stripped = answer, False
    truncated, was_truncated = utils.truncate_with_metadata(stripped_text, max_len=cap)
    return truncated, was_truncated, thinking_stripped


@pytest.mark.parametrize(
    ("answer", "cap", "expected"),
    [
        (
            f"<think>{'so let me think through this... ' * 400}</think>\n\n"
            "The capital of France is Paris.",
            1024,
            "The capital of France is Paris.",
        ),
        (
            f"<think>{'step 1... ' * 500}{utils.VLLM_REASONING_END_STR}"
            "Final answer: 42.",
            256,
            "Final answer: 42.",
        ),
        (
            f"{'leftover reasoning fragment ' * 100}</think>\nAnswer: yes.",
            512,
            "Answer: yes.",
        ),
    ],
)
def test_strip_then_cap_drops_reasoning_prefixes(answer: str, cap: int, expected: str):
    truncated, was_truncated, thinking_stripped = _strip_then_cap(answer, cap)

    assert thinking_stripped is True
    assert was_truncated is False
    assert truncated == expected
    assert "<think>" not in truncated
    assert "</think>" not in truncated


def test_strip_then_cap_passthrough_without_thinking_tags():
    visible = "Paris is the capital of France. " * 50

    truncated, was_truncated, thinking_stripped = _strip_then_cap(visible, cap=512)

    assert thinking_stripped is False
    assert was_truncated is True
    assert truncated == visible[:512]


def test_strip_then_cap_unclosed_think_block_remains_truncated():
    answer = f"<think>{'still reasoning ' * 1000}"

    truncated, was_truncated, thinking_stripped = _strip_then_cap(answer, cap=256)

    assert thinking_stripped is False
    assert was_truncated is True
    assert truncated.startswith("<think>")


def test_strip_then_cap_disabled_preserves_pre_fix_behavior():
    answer = f"<think>{'deep thinking ' * 400}</think>\nShort answer."

    truncated, was_truncated, thinking_stripped = _strip_then_cap(
        answer, cap=1024, strip=False
    )

    assert thinking_stripped is False
    assert was_truncated is True
    assert truncated.startswith("<think>")
    assert "</think>" not in truncated


def test_make_model_openrouter_strips_vllm_only_kwargs(monkeypatch):
    """vLLM-engine-only kwargs must not leak into ChatOpenAI.model_kwargs.

    Regression guard for #20: unknown kwargs forwarded to ``ChatOpenAI`` land
    in ``model_kwargs`` and are then sent to ``chat.completions.create``,
    which rejects them with ``TypeError: unexpected keyword argument
    'max_model_len'``.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    model = make_model(
        "OpenRouter/google/gemma-3-4b-it",
        max_tokens=16,
        max_model_len=4096,
        chat_template="<ct>",
        language_model_only=True,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        temperature=0.5,
    )

    assert "max_model_len" not in model.model_kwargs
    assert "chat_template" not in model.model_kwargs
    assert "language_model_only" not in model.model_kwargs
    assert "gpu_memory_utilization" not in model.model_kwargs
    assert "enforce_eager" not in model.model_kwargs
    assert model.max_tokens == 16
    assert model.temperature == 0.5


def test_init_llm_with_retry_recovers_from_transient_cuda_error(monkeypatch):
    monkeypatch.setattr(utils, "_VLLM_INIT_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(utils, "_VLLM_INIT_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(utils.time, "sleep", lambda *_a, **_k: None)

    calls: list[dict] = []

    def fake_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RuntimeError(
                "CUDA error: CUDA-capable device(s) is/are busy or unavailable\n"
                "Search for 'cudaErrorDevicesUnavailable' ..."
            )
        return "llm"

    result = utils._init_llm_with_retry(fake_llm, model="m", trust_remote_code=True)
    assert result == "llm"
    assert len(calls) == 3


def test_init_llm_with_retry_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(utils, "_VLLM_INIT_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(utils, "_VLLM_INIT_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(utils.time, "sleep", lambda *_a, **_k: None)

    def always_fails(**_kwargs):
        raise RuntimeError("cudaErrorDevicesUnavailable")

    with pytest.raises(RuntimeError, match="cudaErrorDevicesUnavailable"):
        utils._init_llm_with_retry(always_fails, model="m")


def test_init_llm_with_retry_reraises_non_matching_errors_immediately(monkeypatch):
    monkeypatch.setattr(utils, "_VLLM_INIT_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(utils, "_VLLM_INIT_BACKOFF_SECONDS", 0)

    call_count = 0

    def fails_once(**_kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("bad config")

    with pytest.raises(ValueError, match="bad config"):
        utils._init_llm_with_retry(fails_once, model="m")
    assert call_count == 1


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: unknown error",
        "NCCL error",
    ],
)
def test_init_llm_with_retry_does_not_retry_broad_runtime_errors(monkeypatch, message):
    monkeypatch.setattr(utils, "_VLLM_INIT_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(utils, "_VLLM_INIT_BACKOFF_SECONDS", 0)

    call_count = 0

    def fails_once(**_kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match=message):
        utils._init_llm_with_retry(fails_once, model="m")
    assert call_count == 1
