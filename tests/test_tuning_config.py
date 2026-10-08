"""Native NePS configuration and search-space contracts."""

import pytest
from pydantic import ValidationError

from judgearena.config import TuneJudgeArgs
from judgearena.tuning.search_space import (
    FIDELITY,
    apply_overrides,
    axis_overrides,
    build_neps_space,
    decode_config,
)


def _settings():
    return TuneJudgeArgs(
        search_space={
            "judge": {
                "temperature": {"lower": 0.0, "upper": 1.0, "prior_confidence": "low"},
                "model": {
                    "choices": ["Dummy/judge", "Dummy/other"],
                    "prior": "Dummy/other",
                },
                "max_out_tokens": {"lower": 16, "upper": 32, "log": True},
            }
        },
        optimizer={"name": "neps_priorband", "eta": 3},
        fidelity={"battles_per_model": {"lower": 1, "upper": 3}},
        neps={"total_evaluations_to_spend": 10},
        objectives=["agreement"],
    )


def test_nested_domains_and_value_priors():
    from judgearena.tuning.search_space import parameter_specs

    neps = pytest.importorskip("neps")
    settings = _settings()
    specs = parameter_specs(
        settings.search_space, {"judge": {"temperature": 0.2}}, settings.fidelity
    )
    attrs = build_neps_space(specs).get_attrs()
    assert attrs["judge.model"].prior == 1
    assert attrs["judge.model"].choices == ("Dummy/judge", "Dummy/other")
    assert isinstance(attrs["judge.temperature"], neps.Float)
    assert attrs["judge.temperature"].prior == 0.2
    assert isinstance(attrs["judge.max_out_tokens"], neps.Integer)
    assert attrs["judge.max_out_tokens"].log
    assert isinstance(attrs[FIDELITY], neps.IntegerFidelity)
    assert {"judge.temperature": 1.0} in list(axis_overrides(specs))
    overrides, battles = decode_config({"judge.temperature": 0.37, FIDELITY: 3})
    base = {"judge": {"temperature": 0.2}}
    assert apply_overrides(base, overrides)["judge"]["temperature"] == 0.37
    assert base["judge"]["temperature"] == 0.2 and battles == 3
    with pytest.raises(ValueError, match="tuner-owned"):
        parameter_specs({"meta_eval": {"split": ["test"]}}, base, settings.fidelity)


def test_task_defaults_and_partial_overrides():
    from judgearena.config import RunConfig
    from judgearena.tasks.registry import get_packaged_task

    task = get_packaged_task("tune-judge")
    assert task.spec.dataset is None
    cfg = RunConfig(
        task="tune-judge",
        judge={"model": "Dummy/judge"},
        tune_judge={
            "search_space": {"judge": {"temperature": [0.0, 1.0]}},
            "fidelity": {"battles_per_model": {"lower": 30}},
            "optimizer": {"eta": 2},
            "neps": {"total_evaluations_to_spend": 6},
        },
    )
    assert cfg.tune_judge.meta_eval_task == "meta-eval-lmarena-140k-en"
    assert cfg.tune_judge.objectives == ["agreement", "cost"]
    assert cfg.tune_judge.fidelity == {"battles_per_model": {"lower": 30, "upper": 90}}
    assert cfg.tune_judge.optimizer == {
        "name": "mo_hyperband",
        "eta": 2,
        "mo_selector": "epsnet",
    }
    assert cfg.tune_judge.neps["ignore_errors"] is True


@pytest.mark.parametrize(
    "change,match",
    [
        ({"neps": {"total_cost_to_spend": 1}}, "managed"),
        ({"neps": {"root_directory": "elsewhere"}}, "managed"),
        ({"neps": {}}, "finite global budget"),
        ({"search_only": True, "run_dir": None}, "explicit"),
        ({"neps": {"worker_evaluations_to_spend": 1}}, "Worker-local"),
        ({"objectives": ["agreement", "cost"]}, "single objective"),
        ({"optimizer": {"name": "mo_hyperband"}}, "agreement and cost"),
        ({"price_per_million_tokens": {"Dummy/judge": -1}}, "nonnegative"),
    ],
)
def test_tuning_config_validation(change, match):
    values = _settings().model_dump()
    if "neps" in change and change["neps"]:
        change = {**change, "neps": {**values["neps"], **change["neps"]}}
    values.update(change)
    with pytest.raises(ValidationError, match=match):
        TuneJudgeArgs(**values)
