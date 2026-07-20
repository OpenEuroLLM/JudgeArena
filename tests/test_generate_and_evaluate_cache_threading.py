from __future__ import annotations

import pandas as pd
import pytest

import judgearena.evaluate as evaluate_module
import judgearena.generate as generate_module
import judgearena.generate_and_evaluate as gae
import judgearena.models as models_module
import judgearena.mt_bench.mt_bench_utils as mt_bench_utils
import judgearena.mt_bench.pairwise_judging as mt_pairwise
from judgearena.config import RunConfig
from judgearena.generate_and_evaluate import main as main_generate_and_eval


def _synthetic_instructions(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {"instruction": [f"Synthetic instruction {i}" for i in range(n)]},
        index=pd.Index(range(n), name="instruction_index"),
    )


def _cfg_with_cache(tmp_path, **overrides) -> RunConfig:
    payload = {
        "task": "alpaca-eval",
        "model": {"name": "Dummy/gen-a", "baseline": "Dummy/gen-b"},
        "judge": {"model": "Dummy/score A: 0 score B: 10", "swap_mode": "fixed"},
        "generation": {"n_instructions": 2},
        "run": {"result_folder": str(tmp_path / "results"), "no_log_file": True},
        "cache": {"store_root": str(tmp_path / "cache")},
    }
    payload.update(overrides)
    return RunConfig(**payload)


@pytest.fixture
def mock_gae_inputs(monkeypatch):
    instructions = _synthetic_instructions()

    monkeypatch.setattr(
        gae,
        "load_instructions",
        lambda dataset, n_instructions=None: (
            instructions.head(n_instructions)
            if n_instructions is not None
            else instructions
        ),
    )
    monkeypatch.setattr(
        gae,
        "try_load_dataset_completions",
        lambda dataset, model, n_instructions: None,
    )


def test_gae_uses_one_shared_cache_handle(mock_gae_inputs, monkeypatch, tmp_path):
    captured: list[tuple[str, object]] = []
    real_gen = generate_module.do_inference
    real_eval = evaluate_module.do_inference

    def spy_gen(*args, **kwargs):
        cache = kwargs.get("cache")
        if cache is not None:
            captured.append(("gen", cache))
        return real_gen(*args, **kwargs)

    def spy_eval(*args, **kwargs):
        cache = kwargs.get("cache")
        if cache is not None:
            captured.append(("eval", cache))
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(generate_module, "do_inference", spy_gen)
    monkeypatch.setattr(evaluate_module, "do_inference", spy_eval)

    main_generate_and_eval(_cfg_with_cache(tmp_path))

    assert captured
    assert len({id(cache) for _, cache in captured}) == 1
    assert any(role == "gen" for role, _ in captured)
    assert any(role == "eval" for role, _ in captured)


def test_mt_bench_uses_one_shared_cache_handle(monkeypatch, tmp_path):
    questions = pd.DataFrame(
        {
            "category": ["writing"],
            "turn_1": ["Question 1"],
            "turn_2": ["Question 2"],
        },
        index=pd.Index([1], name="instruction_index"),
    )
    monkeypatch.setattr(
        mt_bench_utils,
        "load_instructions",
        lambda dataset, n_instructions=None: questions,
    )
    monkeypatch.setattr(
        mt_bench_utils,
        "load_mt_bench_model_answers",
        lambda model, n_instructions=None: None,
    )
    captured: list[tuple[str, object]] = []
    real_generation = generate_module.do_inference
    real_judging = mt_pairwise.do_inference

    def spy_generation(*args, **kwargs):
        cache = kwargs.get("cache")
        if cache is not None:
            captured.append(("generation", cache))
        return real_generation(*args, **kwargs)

    def spy_judging(*args, **kwargs):
        cache = kwargs.get("cache")
        if cache is not None:
            captured.append(("judging", cache))
        return real_judging(*args, **kwargs)

    monkeypatch.setattr(generate_module, "do_inference", spy_generation)
    monkeypatch.setattr(mt_pairwise, "do_inference", spy_judging)
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "Dummy/gen-a", "baseline": "Dummy/gen-b"},
        judge={"model": "Dummy/[[A]]", "swap_mode": "fixed"},
        generation={"n_instructions": 1},
        run={"result_folder": str(tmp_path / "results"), "no_log_file": True},
        cache={"store_root": str(tmp_path / "cache")},
    )

    main_generate_and_eval(cfg)

    assert {role for role, _ in captured} == {"generation", "judging"}
    assert len({id(cache) for _, cache in captured}) == 1


def test_gae_second_run_reuses_cached_rows(mock_gae_inputs, monkeypatch, tmp_path):
    uncached_calls = {"count": 0}
    real_uncached = models_module._do_inference_uncached

    def counting_uncached(*args, **kwargs):
        uncached_calls["count"] += 1
        return real_uncached(*args, **kwargs)

    monkeypatch.setattr(models_module, "_do_inference_uncached", counting_uncached)

    cfg = _cfg_with_cache(tmp_path)
    prefs_first = main_generate_and_eval(cfg)
    assert uncached_calls["count"] > 0

    uncached_calls["count"] = 0
    prefs_second = main_generate_and_eval(cfg)
    assert uncached_calls["count"] == 0
    assert prefs_second.tolist() == prefs_first.tolist()


def test_gae_preloaded_completions_bypass_generation(
    mock_gae_inputs, monkeypatch, tmp_path
):
    preloaded = pd.DataFrame(
        {
            "completion": ["preloaded-a", "preloaded-b"],
            "instruction_index": [0, 1],
        }
    )
    generation_calls: list[str] = []
    real_gen = models_module._do_inference_uncached

    def track_generation(chat_model, inputs, **kwargs):
        model_spec = getattr(chat_model, "model_spec", None) or getattr(
            chat_model, "name", "unknown"
        )
        generation_calls.append(str(model_spec))
        return real_gen(chat_model, inputs, **kwargs)

    def load_preloaded(dataset, model, n_instructions):
        if model == "Dummy/gen-a":
            return preloaded
        return None

    monkeypatch.setattr(gae, "try_load_dataset_completions", load_preloaded)
    monkeypatch.setattr(models_module, "_do_inference_uncached", track_generation)

    main_generate_and_eval(_cfg_with_cache(tmp_path))

    assert not any("gen-a" in call for call in generation_calls)
    assert any("gen-b" in call for call in generation_calls)


def test_gae_judge_row_metadata_includes_models_and_instruction_index(
    mock_gae_inputs, monkeypatch, tmp_path
):
    captured_metadata: list[dict] = []
    real_eval = evaluate_module.do_inference

    def spy_eval(*args, **kwargs):
        cache_meta = kwargs.get("cache_meta")
        if cache_meta is not None:
            captured_metadata.extend(cache_meta.get("metadata", []))
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "do_inference", spy_eval)

    main_generate_and_eval(
        _cfg_with_cache(
            tmp_path,
            model={"name": "Dummy/gen-a", "baseline": "Dummy/gen-b"},
        )
    )

    assert captured_metadata
    first = captured_metadata[0]
    assert first["instruction_index"] == "0"
    assert first["model_A"] == "Dummy/gen-a"
    assert first["model_B"] == "Dummy/gen-b"
    assert first["orientation"] == "direct"
