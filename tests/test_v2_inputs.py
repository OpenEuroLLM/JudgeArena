"""Focused contracts for v2 experiment inputs."""

from __future__ import annotations

import hashlib
import json
import math

import pandas as pd
import pytest

from judgearena.benchmarks.meta_eval import runner as meta_runner
from judgearena.config import RunConfig
from judgearena.tuning import runner as tune_runner
from judgearena.tuning import session

_TUNE = {
    "search_space": {"judge": {"temperature": [0.0, 1.0]}},
    "neps": {"total_evaluations_to_spend": 1},
}


def test_objective_bounds_normalize_hash_and_preserve_raw():
    bad = {**_TUNE, "objective_bounds": {"agreement": [0.9, 0.5]}}
    with pytest.raises(ValueError, match="finite increasing"):
        RunConfig(task="tune-judge", judge={"model": "Dummy/judge"}, tune_judge=bad)
    bounds = {"agreement": (0.5, 0.9), "cost": (0.0, 10.0)}
    cfg = RunConfig(
        task="tune-judge",
        judge={"model": "Dummy/judge"},
        tune_judge={**_TUNE, "objective_bounds": bounds},
    )
    record = {"status": "completed", "agreement": 0.7, "cost_per_1k_battles": 2.5}
    out = tune_runner._objective(record, ["agreement", "cost"], bounds)
    assert out["objective_to_minimize"] == pytest.approx([0.5, 0.25])
    assert out["info_dict"] is record
    v1 = tune_runner._objective(record, ["agreement", "cost"], {})
    assert v1["objective_to_minimize"] == pytest.approx([0.3, 2.5])
    fail = tune_runner._objective({"status": "failed"}, ["agreement"], bounds)
    assert math.isinf(fail["objective_to_minimize"])
    other = RunConfig(
        task="tune-judge",
        judge={"model": "Dummy/judge"},
        tune_judge={**_TUNE, "objective_bounds": {"agreement": [0.4, 0.9]}},
    )
    assert session._config_hash(cfg) != session._config_hash(other)


def test_panel_file_order_hash_and_judge_kwarg(tmp_path):
    rows = [("a:1", "m1", "m2"), ("a:2", "m2", "m3"), ("a:3", "m1", "m3")]
    pool = pd.DataFrame(rows, columns=["battle_id", "model_a", "model_b"])
    path = tmp_path / "panel.json"
    path.write_text(json.dumps({"validation": {"2": ["a:3", "a:1"]}}))
    ids, digest = meta_runner._load_panel_battle_ids(
        path, split="validation", battles_per_model=2
    )
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    ordered = meta_runner._select_panel_rows(pool, ids, context="Explicit")
    assert ordered["battle_id"].tolist() == ["a:3", "a:1"]
    with pytest.raises(Exception, match="Malformed"):
        meta_runner._load_panel_battle_ids(path, split="test", battles_per_model=2)
    cfg = RunConfig(task="tune-judge", judge={"model": "Dummy/judge"}, tune_judge=_TUNE)
    cfg.meta_eval.panel_file = path
    before = session._config_hash(cfg)
    path.write_text(json.dumps({"validation": {"2": ["a:1", "a:3"]}}))
    assert session._config_hash(cfg) != before
    assert "judge_chat_model" in meta_runner.run_meta_eval.__kwdefaults__
