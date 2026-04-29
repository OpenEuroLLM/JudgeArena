import asyncio
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


def _build_inflight_tracking_chat_model(*, hold_seconds: float = 0.05):
    """Helper: mock chat model whose `ainvoke` records peak concurrent in-flight calls."""

    state = {"in_flight": 0, "peak": 0}

    async def fake_ainvoke(input_item, **_kwargs):
        state["in_flight"] += 1
        state["peak"] = max(state["peak"], state["in_flight"])
        try:
            await asyncio.sleep(hold_seconds)
            return SimpleNamespace(
                content=f"out-{input_item}",
                response_metadata={"finish_reason": "stop"},
            )
        finally:
            state["in_flight"] -= 1

    return SimpleNamespace(ainvoke=fake_ainvoke), state


def test_do_inference_async_path_respects_concurrency_cap(monkeypatch):
    """With JUDGEARENA_JUDGE_MAX_CONCURRENCY=4 and 16 inputs, peak in-flight must stay <= 4."""
    monkeypatch.setenv("JUDGEARENA_JUDGE_MAX_CONCURRENCY", "4")
    chat_model, state = _build_inflight_tracking_chat_model()

    inputs = [f"prompt-{i}" for i in range(16)]
    results = utils.do_inference(
        chat_model=chat_model,
        inputs=inputs,
        use_tqdm=True,
    )

    assert len(results) == 16
    assert state["peak"] <= 4, (
        f"Concurrency cap violated: peak in-flight={state['peak']}, expected <= 4"
    )
    assert state["peak"] >= 1


def test_do_inference_async_path_unbounded_when_env_unset(monkeypatch):
    """Without JUDGEARENA_JUDGE_MAX_CONCURRENCY set, all 16 calls fire concurrently."""
    monkeypatch.delenv("JUDGEARENA_JUDGE_MAX_CONCURRENCY", raising=False)
    chat_model, state = _build_inflight_tracking_chat_model()

    inputs = [f"prompt-{i}" for i in range(16)]
    results = utils.do_inference(
        chat_model=chat_model,
        inputs=inputs,
        use_tqdm=True,
    )

    assert len(results) == 16
    assert state["peak"] > 4, (
        f"Expected unbounded concurrency to overshoot the capped variant; got peak={state['peak']}"
    )


def test_do_inference_async_path_zero_cap_is_unbounded(monkeypatch):
    """JUDGEARENA_JUDGE_MAX_CONCURRENCY=0 falls back to unbounded (defensive default)."""
    monkeypatch.setenv("JUDGEARENA_JUDGE_MAX_CONCURRENCY", "0")
    chat_model, state = _build_inflight_tracking_chat_model()

    inputs = [f"prompt-{i}" for i in range(16)]
    results = utils.do_inference(
        chat_model=chat_model,
        inputs=inputs,
        use_tqdm=True,
    )

    assert len(results) == 16
    assert state["peak"] > 4


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
        temperature=0.5,
    )

    assert "max_model_len" not in model.model_kwargs
    assert "chat_template" not in model.model_kwargs
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
