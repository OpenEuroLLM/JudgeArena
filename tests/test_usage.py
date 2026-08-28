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

    def __init__(self, responses):
        self.responses = responses

    def batch(self, inputs, **kwargs):
        return self.responses[: len(inputs)]


def _provider_message() -> AIMessage:
    return AIMessage(
        content="answer",
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "input_token_details": {"cache_read": 20},
            "output_token_details": {"reasoning": 10},
        },
        response_metadata={
            "id": "gen-test",
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
    )


def test_do_inference_collects_provider_usage_without_changing_text_results():
    with track_usage() as tracker:
        outputs = do_inference(
            FakeModel([_provider_message()]),
            ["prompt"],
            stage="judging",
        )
        snapshot = tracker.snapshot()

    assert outputs == ["answer"]
    assert len(snapshot.requests) == 1
    usage = snapshot.requests[0]
    assert usage.stage == "judging"
    assert usage.model == "google/test-model"
    assert usage.input_tokens == 120
    assert usage.output_tokens == 30
    assert usage.reasoning_tokens == 10
    assert usage.cached_tokens == 20
    assert usage.cost_usd == pytest.approx(0.00125)


def test_do_inference_collects_canonical_langchain_usage_metadata():
    message = AIMessage(
        content="answer",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "input_token_details": {"cache_read": 3},
            "output_token_details": {"reasoning": 1},
        },
        response_metadata={"model_name": "openai/responses-model"},
    )

    with track_usage() as tracker:
        do_inference(FakeModel([message]), ["prompt"], stage="judging")
        usage = tracker.snapshot().requests[0]

    assert usage.input_tokens == 10
    assert usage.output_tokens == 2
    assert usage.total_tokens == 12
    assert usage.cached_tokens == 3
    assert usage.reasoning_tokens == 1


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


def test_canonical_usage_survives_malformed_raw_field_values():
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
                "total_tokens": float("nan"),
                "prompt_tokens_details": {"cached_tokens": True},
                "completion_tokens_details": {"reasoning_tokens": float("inf")},
                "cost": "unknown",
            },
        },
    )

    with track_usage() as tracker:
        do_inference(FakeModel([message]), ["prompt"], stage="judging")
        usage = tracker.snapshot().requests[0]

    assert usage.input_tokens == 101
    assert usage.output_tokens == 23
    assert usage.total_tokens == 124
    assert usage.cached_tokens == 17
    assert usage.reasoning_tokens == 9
    assert usage.cost_usd == pytest.approx(0.125)


def test_do_inference_keeps_usage_on_structured_results():
    with track_usage():
        outputs = do_inference(
            FakeModel([_provider_message()]),
            ["prompt"],
            return_top_logprobs=True,
            stage="judging",
        )

    assert isinstance(outputs[0], InferenceResult)
    assert outputs[0].text == "answer"
    assert outputs[0].usage is not None
    assert outputs[0].usage.total_tokens == 150


def test_run_usage_reports_partial_cost_without_presenting_it_as_complete():
    usage = RunUsage(
        requests=(
            RequestUsage(
                stage="generation",
                model="candidate",
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cost_usd=0.1,
            ),
            RequestUsage(
                stage="judging",
                model="local-judge",
                input_tokens=20,
                output_tokens=2,
                total_tokens=22,
            ),
        )
    ).to_dict()

    assert usage["total"] == {
        "requests": 2,
        "input_tokens": 30,
        "output_tokens": 7,
        "total_tokens": 37,
        "reasoning_tokens": None,
        "cached_tokens": None,
        "cost_usd": pytest.approx(0.1),
        "requests_with_any_token_usage": 2,
        "requests_with_input_tokens": 2,
        "requests_with_output_tokens": 2,
        "requests_with_total_tokens": 2,
        "requests_with_reasoning_tokens": 0,
        "requests_with_cached_tokens": 0,
        "requests_with_cost": 1,
    }
    assert usage["scope"] == (
        "successful_model_responses_returned_by_completed_inference_calls"
    )
    assert usage["source"] == "provider_response_metadata_when_available"
    assert set(usage["by_stage"]) == {"generation", "judging"}
    assert set(usage["by_model"]) == {"candidate", "local-judge"}


def test_run_usage_reports_per_field_partial_token_coverage():
    usage = RunUsage(
        requests=(
            RequestUsage(stage="generation", input_tokens=10),
            RequestUsage(stage="generation", output_tokens=5),
        )
    )
    summary = usage.summary()

    assert summary["input_tokens"] == 10
    assert summary["output_tokens"] == 5
    assert summary["requests_with_input_tokens"] == 1
    assert summary["requests_with_output_tokens"] == 1
    assert "partial: input 1/2, output 1/2" in usage.format_summary()


def test_batch_usage_log_marks_missing_responses_as_partial(caplog):
    caplog.set_level("INFO", logger="judgearena")
    with track_usage():
        do_inference(
            FakeModel([_provider_message(), "no metadata"]),
            ["one", "two"],
            stage="judging",
        )

    assert "partial: input 1/2, output 1/2" in caplog.text
    assert "$0.001250 reported (partial)" in caplog.text


def test_write_run_metadata_includes_active_run_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata_module, "_get_dependency_versions", lambda **_: {})
    monkeypatch.setattr(metadata_module, "_get_git_hash", lambda **_: None)

    with track_usage():
        record_usage(
            [
                RequestUsage(
                    stage="judging",
                    model="judge",
                    input_tokens=10,
                    output_tokens=1,
                    total_tokens=11,
                    cost_usd=0.002,
                )
            ]
        )
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
        record_usage(
            [
                RequestUsage(
                    stage="judging",
                    model="judge",
                    input_tokens=4,
                    output_tokens=1,
                    total_tokens=5,
                    cost_usd=0.0001,
                )
            ]
        )
        return "done"

    resolved = SimpleNamespace(
        adapter=SimpleNamespace(name="test", runner=fake_runner),
        task=None,
    )
    monkeypatch.setattr(benchmark_runner, "resolve_benchmark", lambda task: resolved)

    result = benchmark_runner.run_benchmark(SimpleNamespace(task="test"))

    assert result == "done"
    assert "Model usage:" in capsys.readouterr().out
    assert current_run_usage() is None


class DivergentGenerationModel:
    model_name = "test/divergent"

    def __init__(self):
        self.batch_kwargs = None

    def batch(self, *, inputs, **kwargs):
        self.batch_kwargs = kwargs
        return ["batch" for _ in inputs]

    async def ainvoke(self, _input, **_kwargs):
        return "async"


def test_generation_paths_preserve_existing_sync_async_behavior(monkeypatch):
    model = DivergentGenerationModel()
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
    assert model.batch_kwargs == {"max_tokens": 123}


def test_nested_usage_tracking_explicitly_shares_the_outer_run():
    with track_usage() as outer:
        record_usage([RequestUsage(stage="generation", input_tokens=1)])
        with track_usage() as inner:
            assert inner is outer
            record_usage([RequestUsage(stage="judging", output_tokens=1)])
        assert len(outer.snapshot().requests) == 2

    assert current_run_usage() is None
