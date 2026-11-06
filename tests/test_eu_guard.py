import pandas as pd
import pytest

from judgearena.benchmarks.eu_guard import EUGuardMetric
from judgearena.datasets import load_task_data


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


@pytest.mark.parametrize("config", ["jev", "gemma"])
def test_guard_runner_uses_saved_prompts_and_replays(tmp_path, monkeypatch, config):
    from types import SimpleNamespace

    from judgearena.benchmarks.runner import run_benchmark
    from judgearena.config import build_run_config

    source = tmp_path / "test.jsonl"
    pd.DataFrame(
        {
            "example_id": ["x-de"],
            "prompt": ["Frage"],
            "language_code": ["de"],
            "target": ["should_refuse"],
            "concept_id": ["x"],
        }
    ).to_json(source, orient="records", lines=True)
    monkeypatch.setattr(
        "judgearena.datasets.eu_guard.hf_hub_download", lambda **kw: source
    )
    cfg = build_run_config(
        [
            "--config_path",
            f"configs/eu_guard_prompt_{config}.yaml",
            "--run.store_root",
            str(tmp_path / "cache"),
            "--run.result_folder",
            str(tmp_path / "results"),
        ]
    )
    data = load_task_data(cfg.resolved_task)
    assert data.model_outputs["completion"].tolist() == ["Frage"]
    assert load_task_data("eu-guard").model_outputs is None
    monkeypatch.setattr(
        "judgearena.benchmarks.pairwise.comparisons.load_or_generate_completions",
        lambda *a, **kw: pytest.fail("Unexpected generation"),
    )
    raw = (
        '{"answers":{"refusal":{"noul":0.9}}}'
        if config == "jev"
        else '```json\n{"score":0.9}\n```'
    )

    def batch(inputs, **kwargs):
        for value in inputs:
            text = (
                value["state"]["instruction"] if config == "jev" else value.to_string()
            )
            assert "Frage" in text
            assert "should_refuse" not in text
        return [raw] * len(inputs)

    monkeypatch.setattr(
        "judgearena.models.make_model",
        lambda *a, **kw: SimpleNamespace(batch=batch),
    )
    first = run_benchmark(cfg, data=data)
    monkeypatch.setattr(
        "judgearena.models.make_model", lambda *a, **kw: pytest.fail("Cache miss")
    )
    pd.testing.assert_frame_equal(first, run_benchmark(cfg, data=data))
    assert first.iloc[0]["concept_id"] == "x"
    assert EUGuardMetric().calculate(first)["fcr"] == 0


def test_external_guard_labels():
    from scripts.eu_guard import parse_llama_guard

    assert parse_llama_guard("safe\n")["label"] == "answer"
    assert parse_llama_guard("unsafe\nS1")["details"] == {"categories": ["S1"]}
    assert parse_llama_guard("possibly unsafe") is None
    assert parse_llama_guard("") is None
