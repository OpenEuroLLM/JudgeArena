import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import judgearena.artifacts.metadata as metadata_module
import judgearena.benchmarks.runner as benchmark_runner
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
        "usage_reported_requests": 2,
        "cost_reported_requests": 1,
        "usage_status": "complete",
        "cost_status": "partial",
    }
    assert set(usage["by_stage"]) == {"generation", "judging"}
    assert set(usage["by_model"]) == {"candidate", "local-judge"}


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
