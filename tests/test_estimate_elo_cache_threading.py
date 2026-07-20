from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import judgearena.estimate_elo_ratings as estimate_elo_ratings
import judgearena.evaluate as evaluate_module
import judgearena.generate as generate_module
import judgearena.models as models_module
from judgearena.config import RunConfig
from judgearena.estimate_elo_ratings import main


def _make_conversation(content_user: str, content_assistant: str) -> list[dict]:
    return [
        {"role": "user", "content": content_user},
        {"role": "assistant", "content": content_assistant},
    ]


@pytest.fixture
def synthetic_arena_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(30):
        ma, mb = rng.choice(
            ["arena_model_alpha", "arena_model_beta", "arena_model_gamma"],
            size=2,
            replace=False,
        )
        rows.append(
            {
                "question_id": f"q{i}",
                "tstamp": 1700000000 + i,
                "model_a": ma,
                "model_b": mb,
                "winner": rng.choice(["model_a", "model_b", "tie"]),
                "conversation_a": _make_conversation(
                    f"Instruction {i}", f"Response A {i}"
                ),
                "conversation_b": _make_conversation(
                    f"Instruction {i}", f"Response B {i}"
                ),
                "benchmark": "TestArena",
                "lang": rng.choice(["en", "fr"]),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def mock_elo_deps(monkeypatch, synthetic_arena_df):
    monkeypatch.setattr(
        estimate_elo_ratings,
        "load_arena_dataframe",
        lambda arena: synthetic_arena_df,
    )


def _cfg_with_cache(tmp_path, **overrides) -> RunConfig:
    payload = {
        "task": "elo-comparia",
        "model": {"name": "Dummy/my model"},
        "judge": {"model": "Dummy/score A: 0 score B: 10", "swap_mode": "fixed"},
        "generation": {"n_instructions": 5},
        "elo": {"arena": "ComparIA", "n_bootstraps": 2},
        "run": {"result_folder": str(tmp_path / "results"), "no_log_file": True},
        "cache": {"store_root": str(tmp_path / "cache")},
    }
    payload.update(overrides)
    return RunConfig(**payload)


def test_elo_uses_one_shared_cache_handle(monkeypatch, tmp_path):
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

    main(_cfg_with_cache(tmp_path))

    assert captured
    assert len({id(cache) for _, cache in captured}) == 1
    assert any(role == "gen" for role, _ in captured)
    assert any(role == "eval" for role, _ in captured)


def test_elo_second_run_reuses_cached_rows(monkeypatch, tmp_path):
    uncached_calls = {"count": 0}
    real_uncached = models_module._do_inference_uncached

    def counting_uncached(*args, **kwargs):
        uncached_calls["count"] += 1
        return real_uncached(*args, **kwargs)

    monkeypatch.setattr(models_module, "_do_inference_uncached", counting_uncached)

    cfg = _cfg_with_cache(tmp_path)
    first = main(cfg)
    assert uncached_calls["count"] > 0

    uncached_calls["count"] = 0
    second = main(cfg)
    assert uncached_calls["count"] == 0
    assert second["winrate"] == pytest.approx(first["winrate"])


def test_elo_judge_row_metadata_includes_arena_and_battle_fields(monkeypatch, tmp_path):
    captured_metadata: list[dict] = []
    real_eval = evaluate_module.do_inference

    def spy_eval(*args, **kwargs):
        cache_meta = kwargs.get("cache_meta")
        if cache_meta is not None:
            captured_metadata.extend(cache_meta.get("metadata", []))
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "do_inference", spy_eval)

    main(
        _cfg_with_cache(
            tmp_path,
            model={"name": "Dummy/focal-model"},
        )
    )

    assert captured_metadata
    first = captured_metadata[0]
    assert first["arena"] == "ComparIA"
    assert first["source"] == "elo-judge"
    assert first["focal_model"] == "Dummy/focal-model"
    assert first["opponent_model"]
    assert first["position"] in {"A", "B"}
    assert first["question_id"] == "q0"
    assert first["orientation"] == "direct"


def _configure_calibration_arena(monkeypatch, synthetic_arena_df):
    frames = []
    for block in range(20):
        chunk = synthetic_arena_df.copy()
        chunk["question_id"] = [f"q{block * len(chunk) + j}" for j in range(len(chunk))]
        chunk.index = chunk.index + block * len(chunk)
        frames.append(chunk)
    large_arena_df = pd.concat(frames)
    anchor_battles = pd.DataFrame(
        {
            "model_a": ["arena_model_alpha"] * len(large_arena_df),
            "model_b": ["arena_model_beta"] * len(large_arena_df),
            "winner": ["model_a", "model_b"] * (len(large_arena_df) // 2),
            "pref": [0.0, 1.0] * (len(large_arena_df) // 2),
            "pref_hard": [0.0, 1.0] * (len(large_arena_df) // 2),
            "source": ["human"] * len(large_arena_df),
            "question_id": large_arena_df["question_id"].tolist(),
        },
        index=large_arena_df.index,
    )
    monkeypatch.setattr(
        estimate_elo_ratings,
        "arena_anchor_battles",
        lambda _df: anchor_battles,
    )
    monkeypatch.setattr(
        estimate_elo_ratings,
        "load_arena_dataframe",
        lambda arena: large_arena_df,
    )
    return large_arena_df


def test_elo_calibration_reuses_shared_cache(monkeypatch, tmp_path, synthetic_arena_df):
    _configure_calibration_arena(monkeypatch, synthetic_arena_df)

    captured: list[tuple[str, object]] = []
    real_eval = evaluate_module.do_inference

    def spy_eval(*args, **kwargs):
        cache = kwargs.get("cache")
        cache_meta = kwargs.get("cache_meta")
        if cache is not None:
            captured.append(("eval", cache))
        if cache_meta is not None:
            for row in cache_meta.get("metadata", []):
                if row.get("purpose") == "temperature_calibration":
                    captured.append(("cal_meta", row))
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "do_inference", spy_eval)

    main(
        _cfg_with_cache(
            tmp_path,
            elo={
                "arena": "ComparIA",
                "n_bootstraps": 2,
                "calibrate_temperature": True,
                "calibration_size": 3,
            },
        )
    )

    eval_caches = [cache for role, cache in captured if role == "eval"]
    assert eval_caches
    assert len({id(cache) for cache in eval_caches}) == 1
    cal_rows = [row for role, row in captured if role == "cal_meta"]
    assert cal_rows
    assert cal_rows[0]["source"] == "elo-calibration"
    assert cal_rows[0]["question_id"]


def test_elo_cache_hit_reparses_scores_with_recomputed_calibration(
    monkeypatch, tmp_path, synthetic_arena_df
):
    _configure_calibration_arena(monkeypatch, synthetic_arena_df)
    uncached_calls = {"count": 0}
    real_uncached = models_module._do_inference_uncached

    def counting_uncached(*args, **kwargs):
        uncached_calls["count"] += 1
        return real_uncached(*args, **kwargs)

    monkeypatch.setattr(models_module, "_do_inference_uncached", counting_uncached)
    cfg = _cfg_with_cache(
        tmp_path,
        judge={"model": "Dummy/score A: 2 score B: 1", "swap_mode": "fixed"},
        elo={
            "arena": "ComparIA",
            "n_bootstraps": 1,
            "calibrate_temperature": True,
            "calibration_size": 20,
        },
    )

    monkeypatch.setattr(estimate_elo_ratings, "calibrate_temperature", lambda *_: 0.5)
    first = main(cfg)
    assert uncached_calls["count"] > 0

    uncached_calls["count"] = 0
    monkeypatch.setattr(estimate_elo_ratings, "calibrate_temperature", lambda *_: 5.0)
    second = main(cfg)

    assert uncached_calls["count"] == 0
    assert second["elo_mean"] != pytest.approx(first["elo_mean"])


def test_elo_judge_metadata_fallback_without_question_id(monkeypatch, tmp_path):
    arena_no_qid = pd.DataFrame(
        {
            "tstamp": [1700000000],
            "model_a": ["arena_model_alpha"],
            "model_b": ["arena_model_beta"],
            "winner": ["model_a"],
            "conversation_a": [
                [
                    {"role": "user", "content": "Instruction 0"},
                    {"role": "assistant", "content": "Response A 0"},
                ]
            ],
            "conversation_b": [
                [
                    {"role": "user", "content": "Instruction 0"},
                    {"role": "assistant", "content": "Response B 0"},
                ]
            ],
            "benchmark": "TestArena",
            "lang": ["en"],
        }
    )
    monkeypatch.setattr(
        estimate_elo_ratings,
        "load_arena_dataframe",
        lambda arena: arena_no_qid,
    )

    captured_metadata: list[dict] = []
    real_eval = evaluate_module.do_inference

    def spy_eval(*args, **kwargs):
        cache_meta = kwargs.get("cache_meta")
        if cache_meta is not None:
            captured_metadata.extend(cache_meta.get("metadata", []))
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "do_inference", spy_eval)

    main(
        _cfg_with_cache(
            tmp_path,
            generation={"n_instructions": 1},
            elo={"arena": "ComparIA", "n_bootstraps": 1},
        )
    )

    judge_rows = [row for row in captured_metadata if row.get("source") == "elo-judge"]
    assert judge_rows
    assert "question_id" not in judge_rows[0]
    assert judge_rows[0]["battle_identity"]


def test_elo_swap_mode_both_doubles_llm_judged_battles(tmp_path):
    result = main(
        RunConfig(
            task="elo-comparia",
            model={"name": "Dummy/my model"},
            judge={
                "model": "Dummy/score A: 0 score B: 10",
                "swap_mode": "both",
            },
            generation={"n_instructions": 4},
            elo={"arena": "ComparIA", "n_bootstraps": 1},
            run={"result_folder": str(tmp_path), "no_log_file": True},
        )
    )
    assert result["llm_judged_battles"] == 8
    assert result["num_battles"] == 4
