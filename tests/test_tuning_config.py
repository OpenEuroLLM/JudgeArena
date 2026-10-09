import pytest

from judgearena.config import RunConfig
from judgearena.tuning.search_space import FIDELITY, build_neps_space, parameter_specs


def test_nested_space_and_priors():
    neps = pytest.importorskip("neps")
    domains = {
        "judge": {
            "temperature": {"lower": 0.0, "upper": 1.0},
            "model": {"choices": ["Dummy/a", "Dummy/b"], "prior": "Dummy/b"},
        }
    }
    specs = parameter_specs(
        domains,
        {"judge": {"temperature": 0.2}},
        {"battles_per_model": {"lower": 1, "upper": 3}},
    )
    attrs = build_neps_space(specs).get_attrs()
    assert isinstance(attrs["judge.temperature"], neps.Float)
    assert attrs["judge.temperature"].prior == 0.2
    assert attrs["judge.model"].prior == 1
    assert isinstance(attrs[FIDELITY], neps.IntegerFidelity)
    with pytest.raises(ValueError, match="tuner-owned"):
        parameter_specs({"meta_eval": {"split": ["test"]}}, {}, {})


def test_packaged_defaults_and_target_restrictions():
    settings = {
        "search_space": {"judge": {"temperature": [0.0, 1.0]}},
        "optimizer": {"eta": 2},
        "neps": {"total_evaluations_to_spend": 6},
    }
    cfg = RunConfig(
        task="tune-judge", judge={"model": "Dummy/judge"}, tune_judge=settings
    )
    direct = RunConfig(
        task=cfg.tune_judge.meta_eval_task, judge={"model": "Dummy/judge"}
    )
    assert (cfg.judge, cfg.meta_eval) == (direct.judge, direct.meta_eval)
    assert cfg.tune_judge.optimizer == {
        "name": "mo_hyperband",
        "eta": 2,
        "mo_selector": "epsnet",
    }
    assert cfg.tune_judge.objectives == ["agreement", "cost"]
    assert cfg.tune_judge.fidelity == {"battles_per_model": {"lower": 10, "upper": 90}}
    with pytest.raises(ValueError):
        RunConfig(
            task="tune-judge",
            judge={"model": "Dummy/judge"},
            tune_judge={**settings, "meta_eval_task": "mt-bench"},
        )
