import pytest

import judgearena.models as utils_models
import judgearena.utils as utils
import judgearena.utils.io as utils_io
from judgearena.models import make_model
from judgearena.tasks.registry import load_tasks
from judgearena.utils import safe_parse_int


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("8", 8),
        ("0", 0),
        ("-3", -3),
        (None, None),
        ("", None),
        ("   ", None),
        ("abc", None),
        ("1.5", None),
    ],
)
def test_safe_parse_int(monkeypatch, raw, expected):
    var = "JUDGEARENA_TEST_INT"
    if raw is None:
        monkeypatch.delenv(var, raising=False)
    else:
        monkeypatch.setenv(var, raw)
    assert safe_parse_int(var) == expected


def test_download_all_dispatches_registered_tasks(monkeypatch, tmp_path):
    calls: list[tuple[str, str, object]] = []

    monkeypatch.setattr(utils_io, "data_root", tmp_path)
    monkeypatch.setattr(
        utils_io,
        "download_hf",
        lambda name, local_path: calls.append(("hf", name, local_path)),
    )
    utils_io.download_all()

    assert [name for _, name, _ in calls] == list(load_tasks())
    assert {path for _, _, path in calls} == {tmp_path / "tables"}


def test_strip_thinking_tags_removes_full_reasoning_block():
    raw = (
        "<think>so let me think through this carefully</think>\n\n"
        "The capital of France is Paris."
    )

    cleaned, stripped = utils.strip_thinking_tags_with_metadata(raw)

    assert stripped is True
    assert cleaned == "The capital of France is Paris."
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned


def test_strip_thinking_tags_passthrough_without_reasoning():
    visible = "Paris is the capital of France."

    cleaned, stripped = utils.strip_thinking_tags_with_metadata(visible)

    assert stripped is False
    assert cleaned == visible


def test_strip_thinking_tags_keeps_unclosed_reasoning_block():
    answer = "<think>still reasoning and never closing the tag"

    cleaned, stripped = utils.strip_thinking_tags_with_metadata(answer)

    assert stripped is False
    assert cleaned == answer


def test_make_model_openrouter_uses_native_max_tokens(monkeypatch):
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
        top_logprobs=5,
        extra_body={"provider": {"require_parameters": True}},
    )

    payload = model._get_request_payload("test")
    assert "max_model_len" not in model.model_kwargs
    assert "chat_template" not in model.model_kwargs
    assert "language_model_only" not in model.model_kwargs
    assert "gpu_memory_utilization" not in model.model_kwargs
    assert "enforce_eager" not in model.model_kwargs
    assert "max_completion_tokens" not in payload
    assert payload["extra_body"] == {
        "max_tokens": 16,
        "provider": {"require_parameters": True},
    }
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 5
    assert model.temperature == 0.5


def test_empty_first_token_top_logprobs_are_missing():
    from langchain_core.messages import AIMessage

    response = AIMessage(
        content="M",
        response_metadata={"logprobs": {"content": [{"top_logprobs": []}]}},
    )

    assert utils_models._first_token_top_logprobs(response) is None


def test_make_model_rejects_unsupported_top_logprobs_backend():
    with pytest.raises(ValueError, match="LlamaCpp backend does not support"):
        make_model("LlamaCpp/model.gguf", top_logprobs=5)


def test_init_llm_with_retry_recovers_from_transient_cuda_error(monkeypatch):
    monkeypatch.setattr(utils_models, "_VLLM_INIT_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(utils_models, "_VLLM_INIT_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(utils_models.time, "sleep", lambda *_a, **_k: None)

    calls: list[dict] = []

    def fake_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise RuntimeError(
                "CUDA error: CUDA-capable device(s) is/are busy or unavailable\n"
                "Search for 'cudaErrorDevicesUnavailable' ..."
            )
        return "llm"

    result = utils_models._init_llm_with_retry(
        fake_llm, model="m", trust_remote_code=True
    )
    assert result == "llm"
    assert len(calls) == 3


def test_init_llm_with_retry_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(utils_models, "_VLLM_INIT_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(utils_models, "_VLLM_INIT_BACKOFF_SECONDS", 0)
    monkeypatch.setattr(utils_models.time, "sleep", lambda *_a, **_k: None)

    def always_fails(**_kwargs):
        raise RuntimeError("cudaErrorDevicesUnavailable")

    with pytest.raises(RuntimeError, match="cudaErrorDevicesUnavailable"):
        utils_models._init_llm_with_retry(always_fails, model="m")


def test_init_llm_with_retry_reraises_non_matching_errors_immediately(monkeypatch):
    monkeypatch.setattr(utils_models, "_VLLM_INIT_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(utils_models, "_VLLM_INIT_BACKOFF_SECONDS", 0)

    call_count = 0

    def fails_once(**_kwargs):
        nonlocal call_count
        call_count += 1
        raise ValueError("bad config")

    with pytest.raises(ValueError, match="bad config"):
        utils_models._init_llm_with_retry(fails_once, model="m")
    assert call_count == 1


@pytest.mark.parametrize(
    "message",
    [
        "CUDA error: unknown error",
        "NCCL error",
    ],
)
def test_init_llm_with_retry_does_not_retry_broad_runtime_errors(monkeypatch, message):
    monkeypatch.setattr(utils_models, "_VLLM_INIT_MAX_ATTEMPTS", 4)
    monkeypatch.setattr(utils_models, "_VLLM_INIT_BACKOFF_SECONDS", 0)

    call_count = 0

    def fails_once(**_kwargs):
        nonlocal call_count
        call_count += 1
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match=message):
        utils_models._init_llm_with_retry(fails_once, model="m")
    assert call_count == 1
