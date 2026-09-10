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
    ("usage_metadata", "response_metadata", "expected"),
    [
        (
            {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "input_token_details": {"cache_read": 3},
                "output_token_details": {"reasoning": 1},
            },
            {
                "cost": 0.125,
                "token_usage": {
                    "prompt_tokens": 999,
                    "completion_tokens": -1,
                    "prompt_tokens_details": {"cached_tokens": True},
                    "completion_tokens_details": {"reasoning_tokens": float("inf")},
                },
            },
            (10, 2, 12, 1, 3, 0.125),
        ),
        (
            None,
            {
                "token_usage": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                    "prompt_tokens_details": {"cached_tokens": 20},
                    "completion_tokens_details": {"reasoning_tokens": 10},
                    "cost": 0.00125,
                }
            },
            (120, 30, 150, 10, 20, 0.00125),
        ),
        (None, {"token_usage": "not-a-mapping"}, (None, None, None, None, None, None)),
        (
            {"input_tokens": 10**400, "output_tokens": 2, "total_tokens": 10**400},
            {"token_usage": {"prompt_tokens": 10, "total_tokens": 10**400}},
            (10, 2, 12, None, None, None),
        ),
    ],
    ids=["canonical-precedence", "fallback", "malformed", "oversized"],
)
def test_do_inference_collects_optional_provider_usage(
    usage_metadata, response_metadata, expected
):
    message = AIMessage(
        content="answer",
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
    )
    with track_usage() as tracker:
        outputs = do_inference(FakeModel([message]), ["prompt"], stage="judging")
        assert outputs == ["answer"]
    usage = tracker.snapshot().requests[0]
    assert (usage.stage, usage.model) == ("judging", FakeModel.model_name)
    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.reasoning_tokens,
        usage.cached_tokens,
        usage.cost_usd,
    ) == pytest.approx(expected)


def test_run_benchmark_saves_nested_usage_and_cleans_up(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(metadata_module, "_get_dependency_versions", lambda **_: {})
    monkeypatch.setattr(metadata_module, "_get_git_hash", lambda **_: None)

    def fake_runner(cfg, task):
        generation = RequestUsage(
            stage="generation", model="candidate", input_tokens=10, total_tokens=15
        )
        record_usage([generation])
        with track_usage():
            judging = RequestUsage(
                stage="judging", model="judge", output_tokens=5, cost_usd=0.002
            )
            record_usage([judging])
        return metadata_module.write_run_metadata(
            output_dir=tmp_path, entrypoint="test", run={"task": cfg.task}
        )

    resolved = SimpleNamespace(
        adapter=SimpleNamespace(name="test", runner=fake_runner), task=None
    )
    monkeypatch.setattr(benchmark_runner, "resolve_benchmark", lambda task: resolved)

    path = benchmark_runner.run_benchmark(SimpleNamespace(task="unknown"))
    usage = json.loads(path.read_text())["usage"]
    total = usage["total"]
    assert total["requests"] == 2
    assert total["input_tokens"] == 10
    assert total["output_tokens"] == 5
    assert total["total_tokens"] == 15
    assert total["cost_usd"] == pytest.approx(0.002)
    assert total["requests_with_input_tokens"] == 1
    assert total["requests_with_output_tokens"] == 1
    assert total["requests_with_cost"] == 1
    assert set(usage["by_model"]) == {"candidate", "judge"}
    assert usage["by_stage"]["generation"]["requests"] == 1
    assert usage["by_stage"]["judging"]["requests"] == 1
    output = capsys.readouterr().out
    assert "2 successful response(s)" in output
    assert "partial: input 1/2, output 1/2" in output
    assert "$0.002000" in output
    assert current_run_usage() is None


def test_structured_response_preserves_usage_without_token_breakdown(capsys):
    message = AIMessage(
        content="answer",
        response_metadata={
            "model_name": "provider-model",
            "token_usage": {"total_tokens": 12},
        },
    )
    with track_usage() as tracker:
        (result,) = do_inference(
            FakeModel([message]), ["prompt"], return_top_logprobs=True
        )
    assert isinstance(result, InferenceResult)
    assert result.text == "answer"
    assert result.usage == tracker.snapshot().requests[0]
    assert (result.usage.model, result.usage.total_tokens) == ("provider-model", 12)
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    tracker.render_summary()
    assert "input/output token usage unavailable" in capsys.readouterr().out


def test_failed_inference_reports_no_recorded_responses(monkeypatch, capsys):
    def fail_batch(**kwargs):
        raise ValueError("backend failed")

    def fake_runner(cfg, task):
        return do_inference(SimpleNamespace(batch=fail_batch), ["prompt"])

    resolved = SimpleNamespace(
        adapter=SimpleNamespace(name="test", runner=fake_runner), task=None
    )
    monkeypatch.setattr(benchmark_runner, "resolve_benchmark", lambda task: resolved)

    with pytest.raises(ValueError, match="backend failed"):
        benchmark_runner.run_benchmark(SimpleNamespace(task="test"))
    assert current_run_usage() is None
    assert (
        "No successful model responses were recorded during this run."
        in capsys.readouterr().out
    )


def test_generation_paths_preserve_existing_sync_async_behavior(monkeypatch):
    model = FakeModel(["batch", "batch"], async_response="async")
    monkeypatch.setattr(generate_module, "make_model", lambda *args, **kwargs: model)
    instructions = pd.Series(["one", "two"])

    instruction_outputs = generate_module.generate_instructions(
        instructions, "Dummy/model", use_tqdm=True
    )
    assert instruction_outputs["completion"].tolist() == ["async", "async"]

    base_outputs = generate_module.generate_base(
        instructions, "Dummy/model", max_tokens=123, use_tqdm=True
    )
    assert base_outputs["completion"].tolist() == ["batch", "batch"]
    assert model.batch_kwargs["max_tokens"] == 123
