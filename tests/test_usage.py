import json
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

import judgearena.artifacts.metadata as metadata_module
import judgearena.benchmarks.runner as benchmark_runner
import judgearena.generate as generate_module
from judgearena.models import InferenceResult, do_inference
from judgearena.usage import (
    RequestUsage,
    RunUsage,
    current_run_usage,
    record_usage,
    track_usage,
)


class FakeModel:
    model_name = "google/test-model"

    def __init__(self, responses, *, async_response=None):
        self.responses = responses
        self.async_response = async_response
        self.batch_kwargs = None

    def batch(self, *, inputs, **kwargs):
        self.batch_kwargs = kwargs
        return self.responses[: len(inputs)]

    async def ainvoke(self, _input, **_kwargs):
        return self.async_response


@pytest.mark.parametrize(
    ("message", "structured", "expected"),
    [
        (
            AIMessage(
                content="canonical",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "input_token_details": {"cache_read": 3},
                    "output_token_details": {"reasoning": 1},
                },
                response_metadata={"model_name": "google/test-model"},
            ),
            False,
            (10, 2, 12, 1, 3, None),
        ),
        (
            AIMessage(
                content="fallback",
                response_metadata={
                    "model_name": "google/test-model",
                    "token_usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 30,
                        "total_tokens": 150,
                        "prompt_tokens_details": {"cached_tokens": 20},
                        "completion_tokens_details": {"reasoning_tokens": 10},
                        "cost": 0.00125,
                    },
                },
            ),
            True,
            (120, 30, 150, 10, 20, 0.00125),
        ),
    ],
)
def test_do_inference_collects_canonical_and_fallback_usage(
    message, structured, expected
):
    with track_usage() as tracker:
        outputs = do_inference(
            FakeModel([message]),
            ["prompt"],
            return_top_logprobs=structured,
            stage="judging",
        )
        usage = tracker.snapshot().requests[0]

    if structured:
        assert isinstance(outputs[0], InferenceResult)
        assert outputs[0].text == message.content
        assert outputs[0].usage == usage
    else:
        assert outputs == [message.content]
    assert (usage.stage, usage.model) == ("judging", "google/test-model")
    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.reasoning_tokens,
        usage.cached_tokens,
    ) == expected[:5]
    if expected[5] is None:
        assert usage.cost_usd is None
    else:
        assert usage.cost_usd == pytest.approx(expected[5])


def test_do_inference_ignores_malformed_optional_provider_usage():
    message = AIMessage(
        content="answer",
        response_metadata={"token_usage": "not-a-mapping"},
    )

    with track_usage() as tracker:
        assert do_inference(FakeModel([message]), ["prompt"]) == ["answer"]
        usage = tracker.snapshot().requests[0]

    assert usage.input_tokens is None
    assert usage.output_tokens is None


def test_malformed_raw_usage_does_not_override_canonical_usage():
    message = AIMessage(
        content="answer",
        usage_metadata={
            "input_tokens": 101,
            "output_tokens": 23,
            "total_tokens": 124,
            "input_token_details": {"cache_read": 17},
            "output_token_details": {"reasoning": 9},
        },
        response_metadata={
            "model_name": "openai/responses-model",
            "cost": 0.125,
            "token_usage": {
                "prompt_tokens": "not-a-number",
                "completion_tokens": -1,
                "prompt_tokens_details": {"cached_tokens": True},
                "completion_tokens_details": {"reasoning_tokens": float("inf")},
            },
        },
    )

    with track_usage() as tracker:
        do_inference(FakeModel([message]), ["prompt"], stage="judging")
        usage = tracker.snapshot().requests[0]

    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.cached_tokens,
        usage.reasoning_tokens,
        usage.cost_usd,
    ) == (101, 23, 124, 17, 9, pytest.approx(0.125))


def test_run_usage_reports_partial_field_coverage():
    usage = RunUsage(
        requests=(
            RequestUsage(
                stage="generation",
                model="candidate",
                input_tokens=10,
                total_tokens=15,
                cost_usd=0.1,
            ),
            RequestUsage(stage="judging", model="judge", output_tokens=5),
        )
    )
    summary = usage.summary()
    serialized = usage.to_dict()

    assert (summary["input_tokens"], summary["output_tokens"]) == (10, 5)
    assert summary["requests_with_input_tokens"] == 1
    assert summary["requests_with_output_tokens"] == 1
    assert summary["requests_with_cost"] == 1
    assert "partial: input 1/2, output 1/2" in usage.format_summary()
    assert set(serialized["by_stage"]) == {"generation", "judging"}
    assert set(serialized["by_model"]) == {"candidate", "judge"}
    assert serialized["scope"] == (
        "successful_model_responses_returned_by_completed_inference_calls"
    )


def test_write_run_metadata_includes_active_run_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata_module, "_get_dependency_versions", lambda **_: {})
    monkeypatch.setattr(metadata_module, "_get_git_hash", lambda **_: None)

    with track_usage():
        record_usage([RequestUsage(stage="judging", cost_usd=0.002)])
        path = metadata_module.write_run_metadata(
            output_dir=tmp_path,
            entrypoint="test",
            run={"task": "unknown"},
        )

    metadata = json.loads(path.read_text())
    assert metadata["usage"]["total"]["cost_usd"] == pytest.approx(0.002)
    assert metadata["usage"]["by_stage"]["judging"]["requests"] == 1


def test_run_benchmark_scopes_and_prints_usage(monkeypatch, capsys):
    def fake_runner(cfg, task):
        record_usage([RequestUsage(stage="judging", cost_usd=0.0001)])
        return "done"

    resolved = SimpleNamespace(
        adapter=SimpleNamespace(name="test", runner=fake_runner),
        task=None,
    )
    monkeypatch.setattr(benchmark_runner, "resolve_benchmark", lambda task: resolved)

    assert benchmark_runner.run_benchmark(SimpleNamespace(task="test")) == "done"
    assert "Model usage:" in capsys.readouterr().out
    assert current_run_usage() is None


def test_generation_paths_preserve_existing_sync_async_behavior(monkeypatch):
    model = FakeModel(["batch", "batch"], async_response="async")
    monkeypatch.setattr(generate_module, "make_model", lambda *args, **kwargs: model)
    instructions = pd.Series(["one", "two"])

    instruction_outputs = generate_module.generate_instructions(
        instructions,
        "Dummy/model",
        use_tqdm=True,
    )
    assert instruction_outputs["completion"].tolist() == ["async", "async"]

    base_outputs = generate_module.generate_base(
        instructions,
        "Dummy/model",
        max_tokens=123,
        use_tqdm=True,
    )
    assert base_outputs["completion"].tolist() == ["batch", "batch"]
    assert model.batch_kwargs["max_tokens"] == 123


def test_nested_usage_tracking_explicitly_shares_the_outer_run():
    with track_usage() as outer:
        record_usage([RequestUsage(stage="generation", input_tokens=1)])
        with track_usage() as inner:
            assert inner is outer
            record_usage([RequestUsage(stage="judging", output_tokens=1)])
        assert len(outer.snapshot().requests) == 2

    assert current_run_usage() is None
