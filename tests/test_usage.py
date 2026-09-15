import json
from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

import judgearena.artifacts.metadata as metadata_module
import judgearena.benchmarks.runner as benchmark_runner
import judgearena.generate as generate_module
from judgearena.models import do_inference
from judgearena.usage import RequestUsage, current_run_usage, record_usage, track_usage


class FakeModel:
    model_name = "google/test-model"

    def __init__(self, responses):
        self.responses = responses
        self.batch_kwargs = None

    def batch(self, *, inputs, **kwargs):
        self.batch_kwargs = kwargs
        return self.responses[: len(inputs)]

    async def ainvoke(self, _input, **_kwargs):
        return "async"


def test_do_inference_prefers_canonical_usage():
    message = AIMessage(
        content="answer",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        response_metadata={"token_usage": {"prompt_tokens": 999}, "cost": 0.125},
    )
    with track_usage() as tracker:
        assert do_inference(FakeModel([message]), ["prompt"], stage="judging") == [
            "answer"
        ]
    usage = tracker.snapshot().requests[0]
    assert (usage.stage, usage.model) == ("judging", FakeModel.model_name)
    assert usage.input_tokens == 10
    assert usage.cost_usd == pytest.approx(0.125)


def test_structured_response_preserves_partial_fallback_usage(capsys):
    message = AIMessage(
        content="answer",
        response_metadata={"token_usage": {"total_tokens": 12}},
    )
    with track_usage() as tracker:
        (result,) = do_inference(
            FakeModel([message]), ["prompt"], return_top_logprobs=True
        )
    assert result.text == "answer"
    assert result.usage == tracker.snapshot().requests[0]
    assert result.usage.total_tokens == 12
    tracker.render_summary()
    assert "input/output token usage unavailable" in capsys.readouterr().out


def test_run_benchmark_saves_nested_usage_and_cleans_up(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(metadata_module, "_get_dependency_versions", lambda **_: {})
    monkeypatch.setattr(metadata_module, "_get_git_hash", lambda **_: None)

    def fake_runner(cfg, task):
        record_usage(
            [RequestUsage(stage="generation", model="candidate", input_tokens=10)]
        )
        with track_usage():
            record_usage([RequestUsage(stage="judging", model="judge", cost_usd=0.002)])
        return metadata_module.write_run_metadata(
            output_dir=tmp_path, entrypoint="test", run={"task": cfg.task}
        )

    resolved = SimpleNamespace(
        adapter=SimpleNamespace(name="test", runner=fake_runner), task=None
    )
    monkeypatch.setattr(benchmark_runner, "resolve_benchmark", lambda task: resolved)
    path = benchmark_runner.run_benchmark(SimpleNamespace(task="test"))
    usage = json.loads(path.read_text())["usage"]
    assert usage["total"]["requests"] == 2
    assert usage["total"]["cost_usd"] == pytest.approx(0.002)
    assert usage["by_stage"]["generation"]["input_tokens"] == 10
    assert set(usage["by_model"]) == {"candidate", "judge"}
    assert "partial: input 1/2, output 0/2" in capsys.readouterr().out
    assert current_run_usage() is None


def test_failed_inference_cleans_up(monkeypatch, capsys):
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
    assert "No successful model responses" in capsys.readouterr().out


def test_generation_preserves_sync_async_routing(monkeypatch):
    model = FakeModel(["batch"])
    monkeypatch.setattr(generate_module, "make_model", lambda *args, **kwargs: model)
    instructions = pd.Series(["one"])
    instruction_outputs = generate_module.generate_instructions(
        instructions, "Dummy/model", use_tqdm=True
    )
    assert instruction_outputs["completion"].tolist() == ["async"]
    base_outputs = generate_module.generate_base(
        instructions, "Dummy/model", max_tokens=123, use_tqdm=True
    )
    assert base_outputs["completion"].tolist() == ["batch"]
    assert model.batch_kwargs["max_tokens"] == 123
