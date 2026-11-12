"""HelpSteer3 contracts and shared meta-evaluation regressions."""

from pathlib import Path

import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM

import judgearena.models as models
import judgearena.tasks.definitions.helpsteer3.dataset as source
from judgearena.benchmarks.meta_eval.runner import run_meta_eval
from judgearena.benchmarks.meta_eval.sampling import sample_uniform_pairs
from judgearena.config import RunConfig
from judgearena.tasks.registry import get_packaged_task


@pytest.fixture
def examples(tmp_path, monkeypatch):
    context = [
        {"role": "system", "content": "Remember seventeen."},
        {"role": "user", "content": "Earlier context."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "What is the value?"},
    ]
    records = [
        dict(
            context=context,
            response1=a,
            response2=b,
            domain=domain,
            language=language,
            overall_preference=score,
            individual_preference=[{"score": score, "reasoning": "PRIVATE REASON"}],
        )
        for a, b, domain, language, score in [
            ("Seventeen.", "Twelve.", "general", "english", -3),
            ("Un.", "Deux.", "multilingual", "french", 0),
            ("", "print(1)", "code", "python", 1),
        ]
    ]
    adapter = source.HelpSteer3Adapter(
        get_packaged_task("meta-eval-helpsteer3"), tmp_path
    )
    monkeypatch.setattr(source, "snapshot_download", lambda **kw: None)
    monkeypatch.setattr(source.pd, "read_json", lambda *a, **kw: pd.DataFrame(records))
    return adapter, records


def _config(tmp_path, **kwargs):
    return RunConfig(
        task="meta-eval-helpsteer3",
        judge={"model": "Dummy/judge"},
        run={
            "result_folder": str(tmp_path / "results"),
            "store_root": str(tmp_path / "cache"),
            "no_log_file": True,
        },
        **kwargs,
    )


def test_adapter_and_named_output_views(examples):
    adapter, _ = examples
    data = adapter.load_data()
    assert data.battles.reference_pref.tolist() == [0.0, 0.5, 1.0]
    assert data.battles.overall_preference.tolist() == [-3, 0, 1]
    assert data.battles.lang.tolist() == ["en", "fr", "en"]
    assert data.battles.language.tolist() == ["english", "french", "python"]
    assert data.battles.completion_a.iloc[2] == ""
    assert set(data.battles.model_a) == {"helpsteer3/response1"}
    assert set(data.battles.model_b) == {"helpsteer3/response2"}
    assert data.model_output_rows("helpsteer3/response1").completion.tolist() == [
        "Seventeen.",
        "Un.",
        "",
    ]
    assert len(data.select({"lang": ["fr"]}).instructions) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("overall_preference", 4),
        ("overall_preference", True),
        ("context", []),
        ("response1", None),
    ],
)
def test_invalid_source(examples, field, value):
    adapter, records = examples
    records[0][field] = value
    with pytest.raises(ValueError):
        adapter.load_data()


def test_uniform_sample_is_stable_and_keeps_same_model_pairs(examples):
    rows = examples[0].load_data().battles.assign(model_a="same", model_b="same")
    sample = sample_uniform_pairs(rows, n_instructions=2, seed=7)
    assert len(sample) == 2
    pd.testing.assert_frame_equal(
        sample, sample_uniform_pairs(rows.iloc[::-1], n_instructions=2, seed=7)
    )
    assert len(sample_uniform_pairs(rows, n_instructions=None, seed=7)) == 3
    with pytest.raises(ValueError, match="positive"):
        sample_uniform_pairs(rows, n_instructions=0, seed=7)


def test_metric_rejects_incompatible_labels_before_inference(
    examples, tmp_path, monkeypatch
):
    cfg = _config(tmp_path)
    data = examples[0].load_data()
    data.battles["reference_pref"] = 0.25
    monkeypatch.setattr(
        models,
        "make_model",
        lambda *a, **kw: pytest.fail("Must reject before inference"),
    )
    with pytest.raises(ValueError, match="reference_pref"):
        run_meta_eval(cfg, data=data)


def test_meta_eval_preserves_context_and_swaps_answers(examples, tmp_path, monkeypatch):
    adapter, records = examples
    records.append(records[0].copy())
    data = adapter.load_data()
    cfg = _config(tmp_path, generation={"n_instructions": 1})
    cfg.run.filters = {"domain": ["general"]}

    def make_judge(model, **kwargs):
        assert model == cfg.judge.model
        return FakeListLLM(
            responses=["score_A: 9\nscore_B: 1", "score_A: 1\nscore_B: 9"]
        )

    monkeypatch.setattr(models, "make_model", make_judge)
    first = run_meta_eval(cfg, data=data)
    assert first["n_sampled_battles"] == 1
    original = Path(first["result_path"]).parent
    inputs = pd.read_parquet(original / "annotations.parquet").judge_input.tolist()
    assert all(
        "Remember seventeen" in p
        and "Earlier context" in p
        and "PRIVATE REASON" not in p
        for p in inputs
    )
    assert inputs[0].index("Seventeen.") < inputs[0].index("Twelve.")
    assert inputs[1].index("Twelve.") < inputs[1].index("Seventeen.")
    assert first["metrics"]["meta_eval_agreement"]["all"]["accuracy_complete"] == 1.0
    assert "arena" not in first and "top_models" not in first
    cfg.run.result_folder = str(tmp_path / "replay")
    monkeypatch.setattr(
        models,
        "make_model",
        lambda *a, **kw: pytest.fail("Must replay without inference"),
    )
    replay = run_meta_eval(cfg, data=data)
    assert replay["metrics"]["meta_eval_agreement"]["all"]["accuracy_complete"] == 1.0
