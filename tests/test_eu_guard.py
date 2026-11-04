import json
from io import BytesIO

import pandas as pd
import pytest

from judgearena.benchmarks.eu_guard import EUGuardMetric
from judgearena.cache.inference import JudgementInferenceCache
from judgearena.datasets import load_task_data
from judgearena.judging import JudgeRequest, annotation_frame, execute_judging
from judgearena.models import prepare_model
from judgearena.prompts.registry import render_inputs, resolve_judge_prompt


def test_refusal_metrics_labels_probabilities_and_missing():
    rows = pd.DataFrame(
        {
            "reference_label": ["answer", "answer", "refuse", "refuse", "refuse"],
            "parsed": [{"score": p} for p in [0.1, 0.7, 0.8, 0.9, 0.2]],
        }
    )
    result = EUGuardMetric().calculate(rows)
    assert result["fcr"] == pytest.approx(1 / 3)
    assert result["frr"] == 0.5
    assert result["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)
    labels = rows.assign(
        parsed=[
            {"label": x} for x in ["answer", "refuse", "refuse", "refuse", "answer"]
        ]
    )
    assert EUGuardMetric().calculate(labels) == result
    assert EUGuardMetric(threshold=0.85).calculate(rows)["frr"] == 0
    for invalid in [None, {"score": 2}, {"label": "unknown"}]:
        rows.at[0, "parsed"] = invalid
        missing = EUGuardMetric().calculate(rows)
        assert missing["num_missing"] == 1
        assert missing["macro_f1"] is None
    assert EUGuardMetric().calculate(labels.assign(completion=""))["num_valid"] == 0


@pytest.mark.parametrize("preset", ["eu-guard-behavior", "eu-guard-prompt"])
def test_adapter_decisions_and_replay(tmp_path, monkeypatch, preset):
    source = tmp_path / "test.jsonl"
    pd.DataFrame(
        {
            "example_id": ["x-de", "x-en"],
            "prompt": ["Frage", "Question"],
            "language_code": ["de", "en"],
            "category_label": ["test", "test"],
            "target": ["should_refuse", "should_refuse"],
            "concept_id": ["x", "x"],
        }
    ).to_json(source, orient="records", lines=True)
    monkeypatch.setattr(
        "judgearena.datasets.eu_guard.hf_hub_download", lambda **kw: source
    )
    data = load_task_data("eu-guard").select({"lang": ["de"]})
    samples = (
        data.instructions.assign(completion="I cannot help.")
        if preset.endswith("behavior")
        else data.instructions
    )
    prompt = resolve_judge_prompt(preset=preset)
    inputs = render_inputs(prompt, samples.to_dict("records"))
    assert set(inputs[0]["state"]) == (
        {"instruction", "completion"}
        if preset.endswith("behavior")
        else {"instruction"}
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **kw: BytesIO(
            json.dumps({"answers": {"refusal": {"noul": 0.9}}}).encode()
        ),
    )
    model = prepare_model(
        "Decisions/local",
        endpoint="http://localhost/systemone",
        cache=JudgementInferenceCache(tmp_path / "cache", "eu-guard"),
    )
    requests = [
        JudgeRequest(
            str(i),
            value,
            prompt.parser,
            {"instruction_id": str(i), "model_a": "guard", "model_b": None},
        )
        for i, value in zip(samples.index, inputs, strict=True)
    ]
    results = execute_judging(model, requests)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **kw: pytest.fail("Cache miss")
    )
    replay = execute_judging(model, requests)
    annotations = annotation_frame(samples, results, [prompt])
    pd.testing.assert_frame_equal(
        annotations, annotation_frame(samples, replay, [prompt])
    )
    assert annotations.iloc[0]["concept_id"] == "x"
    assert EUGuardMetric().calculate(annotations)["fcr"] == 0


def test_external_guard_labels():
    from scripts.eu_guard import parse_llama_guard

    assert parse_llama_guard("safe\n")["label"] == "answer"
    assert parse_llama_guard("unsafe\nS1")["details"] == {"categories": ["S1"]}
    assert parse_llama_guard("possibly unsafe") is None
    assert parse_llama_guard("") is None
