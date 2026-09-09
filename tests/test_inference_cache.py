from langchain_core.messages import AIMessage

import judgearena.models as models
from judgearena.inference import CompletionInferenceCache, JudgementInferenceCache
from judgearena.models import InferenceResult, do_inference, prepare_model
from judgearena.usage import track_usage


class EchoModel:
    def __init__(self):
        self.calls = []

    def batch(self, inputs, **_kwargs):
        self.calls.append(inputs)
        return [AIMessage(content=f"generated:{item}") for item in inputs]


def test_full_hit_does_not_materialize_model(tmp_path, monkeypatch):
    cache = CompletionInferenceCache(tmp_path, "arena-hard")
    monkeypatch.setattr(models, "make_model", lambda *_args, **_kwargs: EchoModel())
    metadata = [{"instruction_id": "1"}]
    do_inference(
        prepare_model("Dummy/test-model", cache=cache),
        ["prompt"],
        cache_metadata=metadata,
    )

    def fail_if_materialized(*_args, **_kwargs):
        raise AssertionError("cache hit materialized the model")

    monkeypatch.setattr(models, "make_model", fail_if_materialized)
    with track_usage() as tracker:
        outputs = do_inference(
            prepare_model("Dummy/test-model", cache=cache),
            ["prompt"],
            cache_metadata=metadata,
        )

    assert outputs == ["generated:prompt"]
    assert tracker.snapshot().requests == ()


def test_mixed_hits_and_misses_preserve_order(tmp_path, monkeypatch):
    cache = CompletionInferenceCache(tmp_path, "arena-hard")
    first_backend = EchoModel()
    monkeypatch.setattr(models, "make_model", lambda *_args, **_kwargs: first_backend)
    do_inference(
        prepare_model("Dummy/test-model", cache=cache),
        ["hit"],
        cache_metadata=[{"instruction_id": "hit"}],
    )

    backend = EchoModel()
    monkeypatch.setattr(models, "make_model", lambda *_args, **_kwargs: backend)
    outputs = do_inference(
        prepare_model("Dummy/test-model", cache=cache),
        ["miss-a", "hit", "miss-b"],
        cache_metadata=[
            {"instruction_id": "a"},
            {"instruction_id": "hit"},
            {"instruction_id": "b"},
        ],
    )

    assert outputs == ["generated:miss-a", "generated:hit", "generated:miss-b"]
    assert backend.calls == [["miss-a", "miss-b"]]


def test_judgement_hit_preserves_top_logprobs(tmp_path, monkeypatch):
    cache = JudgementInferenceCache(tmp_path, "arena-hard")

    class LogprobModel:
        def batch(self, inputs, **_kwargs):
            return [
                InferenceResult(
                    text="m",
                    first_token_top_logprobs={"m": -0.1, "M": -2.0},
                )
                for _ in inputs
            ]

    monkeypatch.setattr(models, "make_model", lambda *_args, **_kwargs: LogprobModel())
    metadata = [
        {
            "instruction_id": "1",
            "model_a": "candidate",
            "model_b": "baseline",
            "orientation": "direct",
        }
    ]
    first = do_inference(
        prepare_model("Dummy/judge", cache=cache),
        ["judge prompt"],
        return_top_logprobs=True,
        cache_metadata=metadata,
    )
    second = do_inference(
        prepare_model("Dummy/judge", cache=cache),
        ["judge prompt"],
        return_top_logprobs=True,
        cache_metadata=metadata,
    )

    assert second[0].text == first[0].text
    assert second[0].first_token_top_logprobs == first[0].first_token_top_logprobs
    assert second[0].usage is None
