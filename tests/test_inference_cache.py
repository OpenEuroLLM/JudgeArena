import pandas as pd

import judgearena.evaluate as evaluate
import judgearena.generate as generate
import judgearena.inference as inference
import judgearena.utils as utils
from judgearena.inference import InferenceCache
from judgearena.utils import do_inference, prepare_model


class ConstantModel:
    def __init__(self, output):
        self.output = output

    def batch(self, inputs, **_kwargs):
        return [self.output] * len(inputs)


class EchoModel:
    def __init__(self):
        self.calls = []

    def batch(self, inputs, **_kwargs):
        self.calls.append(inputs)
        return [f"generated:{item}" for item in inputs]


def test_generate_and_judge_full_hits_do_not_materialize_models(tmp_path, monkeypatch):
    completion_cache = InferenceCache(tmp_path, "completions", "arena-hard")
    judgement_cache = InferenceCache(tmp_path, "judgements", "arena-hard")
    battle_model = "VLLM/Qwen/Qwen3-8B"
    judge_model = "VLLM/Qwen/Qwen3-32B"
    monkeypatch.setattr(inference.importlib_metadata, "version", lambda _name: "0.10.2")

    def fake_model(model, **_kwargs):
        output = (
            "Candidate answer" if model == battle_model else "score A: 1 score B: 0"
        )
        return ConstantModel(output)

    monkeypatch.setattr(utils, "make_model", fake_model)
    instructions = pd.Series(["Which answer is better?"], index=[1])
    completions = generate.generate_instructions(
        instructions,
        battle_model,
        use_tqdm=False,
        inference_cache=completion_cache,
    )
    metadata = [
        {
            "instruction_id": "1",
            "model_a": "candidate",
            "model_b": "baseline",
            "orientation": "direct",
        }
    ]
    arguments = {
        "instructions": instructions.tolist(),
        "completions_A": completions["completion"].tolist(),
        "completions_B": ["Baseline answer"],
        "inference_cache": judgement_cache,
        "cache_metadata": metadata,
    }

    first = evaluate.annotate_battles(
        judge_chat_model=prepare_model(judge_model),
        **arguments,
    )

    def fail_if_materialized(*_args, **_kwargs):
        raise AssertionError("cache hit materialized the model")

    monkeypatch.setattr(utils, "make_model", fail_if_materialized)
    cached_completions = generate.generate_instructions(
        instructions,
        battle_model,
        use_tqdm=False,
        inference_cache=completion_cache,
    )
    second = evaluate.annotate_battles(
        judge_chat_model=prepare_model(judge_model),
        **arguments,
    )

    assert cached_completions["completion"].tolist() == ["Candidate answer"]
    assert first[0].judge_completion == "score A: 1 score B: 0"
    assert second[0].judge_completion == first[0].judge_completion


def test_mixed_hits_and_misses_preserve_order(tmp_path, monkeypatch):
    cache = InferenceCache(tmp_path, "completions", "arena-hard")
    monkeypatch.setattr(
        utils, "make_model", lambda *_args, **_kwargs: ConstantModel("cached")
    )
    do_inference(
        prepare_model("Dummy/test-model"),
        ["hit"],
        cache=cache,
        cache_metadata=[{"instruction_id": "hit"}],
    )

    backend = EchoModel()
    monkeypatch.setattr(utils, "make_model", lambda *_args, **_kwargs: backend)
    outputs = do_inference(
        prepare_model("Dummy/test-model"),
        ["miss-a", "hit", "miss-b"],
        cache=cache,
        cache_metadata=[
            {"instruction_id": "a"},
            {"instruction_id": "hit"},
            {"instruction_id": "b"},
        ],
    )

    assert outputs == ["generated:miss-a", "cached", "generated:miss-b"]
    assert backend.calls == [["miss-a", "miss-b"]]


def test_vllm_descriptor_contains_resolved_backend_configuration(monkeypatch):
    monkeypatch.setattr(inference.importlib_metadata, "version", lambda _name: "0.10.2")

    model = prepare_model(
        "VLLM/Qwen/Qwen3-8B",
        max_tokens=32,
        max_model_len=4096,
        tensor_parallel_size=2,
    )

    assert model.descriptor["backend_version"] == "0.10.2"
    assert model.descriptor["model_kwargs"] == {
        "max_tokens": 32,
        "max_model_len": 4096,
        "tensor_parallel_size": 2,
        "chat_template": None,
    }
    assert model.descriptor["sampling"] == {"temperature": 0.6, "top_p": 0.95}
    assert prepare_model("OpenRouter/provider/model").descriptor is None
