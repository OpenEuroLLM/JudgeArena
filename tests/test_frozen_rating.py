import numpy as np
import pandas as pd
import pytest

from judgearena.benchmarks.elo.rating import fit_frozen_rating


def test_frozen_fit_orientation_soft_targets_and_group_bootstrap():
    # Two judgments per prompt; reversed slots already use canonical preferences.
    battles = pd.DataFrame(
        {
            "model_a": ["new", "anchor", "new", "anchor"],
            "model_b": ["anchor", "new", "anchor", "new"],
            "pref": [0.2, 0.8, 0.4, 0.6],
            "split_group": ["first", "first", "second", "second"],
        }
    )
    references = {"anchor": 1000.0}
    result = fit_frozen_rating(battles, "new", references, n_bootstraps=100)
    assert result["rating"] == pytest.approx(1000 + 400 * np.log10(0.7 / 0.3))
    assert result["ci_low"] == pytest.approx(1000 + 400 * np.log10(0.6 / 0.4))
    assert result["ci_high"] == pytest.approx(1000 + 400 * np.log10(0.8 / 0.2))
    assert result == fit_frozen_rating(battles, "new", references, n_bootstraps=100)
    assert references == {"anchor": 1000.0}
    hard = fit_frozen_rating(battles, "new", references, hard=True)
    assert hard["rating"] == hard["ci_low"] == hard["ci_high"] == float("inf")
    battles["pref"] = 1 - battles.pref
    assert fit_frozen_rating(battles, "new", references, hard=True)["rating"] == -float(
        "inf"
    )

    varied = battles.iloc[:2].copy()
    varied["model_a"] = "new"
    varied["model_b"] = ["weak", "strong"]
    varied["pref"] = [1 / (1 + 10 ** ((1100 - r) / 400)) for r in [800, 1200]]
    fitted = fit_frozen_rating(varied, "new", {"weak": 800, "strong": 1200})
    assert fitted["rating"] == pytest.approx(1100)


def test_frozen_fit_rejects_invalid_inputs_and_counts_missing():
    battles = pd.DataFrame(
        {
            "model_a": ["new", "new", "new"],
            "model_b": ["anchor"] * 3,
            "pref": [0.5, 0.5, None],
            "split_group": [1, 2, 3],
        }
    )
    result = fit_frozen_rating(battles, "new", {"anchor": 1000}, n_bootstraps=5)
    assert result["rating"] == result["ci_low"] == result["ci_high"] == 1000
    assert result["num_comparisons"] == 2
    assert result["num_missing"] == 1
    for refs in ({}, {"anchor": float("nan")}):
        with pytest.raises(ValueError):
            fit_frozen_rating(battles, "new", refs)
    for column, value in (("pref", 2), ("split_group", None), ("model_a", "other")):
        invalid = battles.copy()
        invalid.loc[0, column] = value
        with pytest.raises(ValueError):
            fit_frozen_rating(invalid, "new", {"anchor": 1000})
    with pytest.raises(ValueError):
        fit_frozen_rating(battles.assign(pref=np.nan), "new", {"anchor": 1000})


def test_ordinary_frozen_scoring_reuses_candidate_fit_and_keeps_references_fixed():
    from judgearena.benchmarks.elo.scoring import BradleyTerryMetric

    references = pd.DataFrame(
        {
            "model_a": ["a"] * 3,
            "model_b": ["b"] * 3,
            "pref": [0.0, 0.5, 1.0],
            "evaluation_model": None,
        }
    )
    candidate = pd.DataFrame(
        {
            "model_a": ["new"] * 4,
            "model_b": ["a", "b"] * 2,
            "pref": [0.2, 0.4, 0.6, 0.5],
            "evaluation_model": "new",
            "question_id": [1, 2, 3, 4],
        }
    )
    battles = pd.concat([references, candidate], ignore_index=True)
    battles["pref_hard"] = (
        (battles.pref > 0.5).astype(float).where(battles.pref != 0.5, 0.5)
    )
    metric = BradleyTerryMetric(fit_mode="frozen", n_bootstraps=20)
    result = metric.calculate(battles, seed=42)
    expected = fit_frozen_rating(
        candidate,
        "new",
        result["reference_ratings"],
        group_col="question_id",
        n_bootstraps=20,
        seed=42,
    )
    assert all(result[key] == value for key, value in expected.items())
    # Separated reference labels give the same fits as older mixed artifacts.
    separated = battles.copy()
    reference_rows = separated.evaluation_model.isna()
    separated["reference_pref"] = separated.pref.where(reference_rows)
    separated.loc[reference_rows, ["pref", "pref_hard"]] = np.nan
    original = separated.copy(deep=True)
    for mode in ("frozen", "joint"):
        for soft in (False, True):
            configured = BradleyTerryMetric(fit_mode=mode, soft=soft, n_bootstraps=20)
            legacy = configured.calculate(battles, seed=42)
            current = configured.calculate(separated, seed=42)
            assert current["ratings"] == legacy["ratings"]
            assert current["rating"] == legacy["rating"]
            if mode == "frozen":
                assert current["ci_low"] == legacy["ci_low"]
                assert current["ci_high"] == legacy["ci_high"]
            else:
                assert current["bootstrap_ratings"] == legacy["bootstrap_ratings"]
    pd.testing.assert_frame_equal(separated, original)
    changed = battles.copy()
    changed.loc[changed.evaluation_model.notna(), "pref"] = 0.1
    assert metric.calculate(changed)["reference_ratings"] == result["reference_ratings"]
    assert (
        BradleyTerryMetric(fit_mode="joint").calculate(battles)["ratings"]
        != result["ratings"]
    )
    point = BradleyTerryMetric(fit_mode="frozen").calculate(battles)
    assert point["rating"] == result["rating"]
    assert np.isnan(point["ci_low"])


def test_ordinary_elo_frozen_and_joint_replay(tmp_path, monkeypatch):
    import json

    from judgearena.benchmarks.runner import run_benchmark
    from judgearena.config import RunConfig, load_config
    from judgearena.datasets.data import PreferenceData
    from judgearena.models import PreparedModel

    data = PreferenceData.from_battles(
        pd.DataFrame(
            {
                "instruction_index": range(6),
                "instruction": "Question",
                "model_a": "a",
                "model_b": "b",
                "completion_a": "A",
                "completion_b": "B",
                "reference_pref": 0.5,
                "lang": "en",
            }
        )
    )
    cfg = RunConfig(
        task="elo-comparia",
        model={"name": "Dummy/answer"},
        judge={"model": "Dummy/[[A=B]]", "prompt": {"preset": "arena-hard"}},
        generation={"n_instructions": 4},
        protocol={
            "scoring": {
                "metrics": [
                    dict(
                        metric="bradley_terry",
                        name=name,
                        parameters=dict(
                            fit_mode="frozen", soft=soft, n_bootstraps=bootstraps
                        ),
                    )
                    for name, soft, bootstraps in (
                        ("bradley_terry", True, 5),
                        ("hard", False, 0),
                    )
                ]
            }
        },
        run={
            "store_root": str(tmp_path / "cache"),
            "result_folder": str(tmp_path / "results"),
            "use_tqdm": False,
            "no_log_file": True,
        },
    )
    manifest = run_benchmark(cfg, data=data, output_dir=tmp_path / "first")
    first = manifest.parent
    results = json.loads(manifest.read_text())["metrics"]
    frozen = results["bradley_terry"]
    assert frozen["fit_mode"] == "frozen"
    assert results["hard"]["n_bootstraps"] == 0
    assert frozen["rating"] == 1000
    assert not (first / "elo_ratings.json").exists()
    monkeypatch.setattr(
        PreparedModel, "materialize", lambda *a, **kw: pytest.fail("cache miss")
    )
    replay = load_config(first / "config.yaml")
    for request in replay.resolved_task.spec.protocol.scoring.metrics:
        expected_bootstraps = 0 if request.name == "hard" else 5
        assert request.parameters["n_bootstraps"] == expected_bootstraps
        assert request.parameters["soft"] is (request.name != "hard")
        assert results[request.name]["n_bootstraps"] == expected_bootstraps
    manifest = run_benchmark(replay, data=data, output_dir=tmp_path / "replay")
    assert json.loads(manifest.read_text())["metrics"] == results
    values = replay.model_dump()
    requests = [
        request.model_dump()
        for request in replay.resolved_task.spec.protocol.scoring.metrics
    ]
    for request in requests:
        request["parameters"].update(fit_mode="joint", reference_min_battles=0)
    values["protocol"] = {"scoring": {"metrics": requests}}
    replay = RunConfig(**values)
    manifest = run_benchmark(replay, data=data, output_dir=tmp_path / "joint")
    joint = json.loads(manifest.read_text())["metrics"]["bradley_terry"]
    assert joint["ratings"] == frozen["ratings"]
