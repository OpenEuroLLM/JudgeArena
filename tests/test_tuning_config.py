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
        meta_eval_task="meta-eval-comparia-fr",
        neps={
            "optimizer": {"name": "neps_priorband", "eta": 3},
            "total_evaluations_to_spend": 10,
            "pipeline_space": {
                "judge.temperature": dict(
                    type="Float",
                    lower=0.0,
                    upper=1.0,
                    prior=0.2,
                    prior_confidence="medium",
                ),
                "judge.model": dict(
                    type="Categorical",
                    choices=["Dummy/judge", "Dummy/other"],
                    prior=1,
                    prior_confidence="medium",
                ),
                FIDELITY: dict(type="IntegerFidelity", lower=1, upper=3),
            },
        },
        objectives=["agreement"],
        price_per_million_tokens={"Dummy/judge": 1.0, "Dummy/other": 1.0},
    )


def test_dotted_domains_and_index_priors():
    neps = pytest.importorskip("neps")
    specs = _settings().neps["pipeline_space"]
    specs["judge.max_out_tokens"] = dict(type="Integer", lower=16, upper=32)
    attrs = build_neps_space(specs).get_attrs()
    assert attrs["judge.model"].prior == 1
    assert attrs["judge.model"].choices == ("Dummy/judge", "Dummy/other")
    assert isinstance(attrs["judge.temperature"], neps.Float)
    assert attrs["judge.temperature"].prior == 0.2
    assert isinstance(attrs["judge.max_out_tokens"], neps.Integer)
    assert isinstance(attrs[FIDELITY], neps.IntegerFidelity)
    assert {"judge.temperature": 1.0} in list(axis_overrides(specs))
    overrides, battles = decode_config(
        {"judge.temperature": 0.37, "judge.model": "Dummy/other", FIDELITY: 3}
    )
    base = {"judge": {"temperature": 0.2}}
    assert apply_overrides(base, overrides)["judge"] == {
        "temperature": 0.37,
        "model": "Dummy/other",
    }
    assert base["judge"]["temperature"] == 0.2 and battles == 3
    for name, spec in [
        ("meta_eval.split", {"type": "Categorical", "choices": ["test"]}),
        ("judge.temperature", {"type": "IntegerFidelity", "lower": 1, "upper": 3}),
    ]:
        with pytest.raises(ValueError, match="tuner-owned|Only"):
            build_neps_space({**specs, name: spec})


@pytest.mark.parametrize(
    "change,match",
    [
        ({"neps": {"total_cost_to_spend": 1}}, "managed"),
        ({"neps": {"root_directory": "elsewhere"}}, "managed"),
        ({"neps": {}}, "finite global budget"),
        ({"search_only": True, "run_dir": None}, "explicit"),
        ({"neps": {"worker_evaluations_to_spend": 1}}, "Worker-local"),
        ({"objectives": ["agreement", "cost_per_1k_battles"]}, "single objective"),
        ({"neps": {"optimizer": {"name": "mo_hyperband"}}}, "agreement and cost"),
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
