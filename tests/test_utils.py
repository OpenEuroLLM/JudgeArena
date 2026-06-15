import pytest

import judgearena.instruction_dataset.mt_bench as mt_bench_mod
import judgearena.utils as utils
from judgearena.utils import make_model, safe_parse_int


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


def test_download_all_dispatches_arena_hard_versions(monkeypatch, tmp_path):
    calls: list[tuple[str, str, object]] = []

    monkeypatch.setattr(utils, "data_root", tmp_path)
    monkeypatch.setattr(
        utils,
        "download_hf",
        lambda name, local_path: calls.append(("hf", name, local_path)),
    )
    monkeypatch.setattr(
        utils,
        "download_arena_hard",
        lambda dataset, local_tables_path: calls.append(
            ("arena", dataset, local_tables_path)
        ),
    )
    monkeypatch.setattr(
        utils,
        "snapshot_download",
        lambda **kwargs: calls.append(
            ("snapshot", kwargs["repo_id"], kwargs["local_dir"])
        ),
    )
    monkeypatch.setattr(
        mt_bench_mod,
        "download_mt_bench",
        lambda local_dir=None: None,
    )

    utils.download_all()

    tables_dir = tmp_path / "tables"
    assert calls[:5] == [
        ("hf", "alpaca-eval", tables_dir),
        ("arena", "arena-hard-v0.1", tables_dir),
        ("arena", "arena-hard-v2.0", tables_dir),
        ("hf", "m-arena-hard-v0.1", tables_dir),
        ("hf", "m-arena-hard-v2.0", tables_dir),
    ]
    assert calls[5] == (
        "snapshot",
        "geoalgo/multilingual-contexts-to-be-completed",
        tmp_path / "contexts",
    )


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
