"""Translate a judge-tuning search space to and from a neps search space."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from judgearena.config import FloatRange, TuneJudgeArgs

if TYPE_CHECKING:
    import neps

TUNER_OWNED_KEYS = frozenset(
    {"task", "meta_eval.battles_per_model", "meta_eval.split", "run.result_folder"}
)
FIDELITY = "battles_per_model"


def choice_overrides(name: str, value: object) -> dict[str, object]:
    """Return the dotted overrides one choice of an axis applies."""
    return dict(value) if isinstance(value, Mapping) else {name: value}


def axis_overrides(tuning: TuneJudgeArgs) -> Iterator[dict[str, object]]:
    """Yield every choice and range bound, rejecting tuner-owned keys."""
    for name, axis in tuning.search_space.items():
        values = [axis.lower, axis.upper] if isinstance(axis, FloatRange) else axis
        for value in values:
            overrides = choice_overrides(name, value)
            owned = sorted(
                key
                for key in overrides
                if key in TUNER_OWNED_KEYS or key.startswith("tune_judge")
            )
            if owned:
                raise ValueError(
                    f"search_space must not override tuner-owned keys: {owned}"
                )
            yield overrides


def _lookup(values: Mapping[str, Any], path: str) -> object:
    for key in path.split("."):
        if not isinstance(values, Mapping) or key not in values:
            return None
        values = values[key]
    return values


def build_neps_space(
    tuning: TuneJudgeArgs, base: Mapping[str, Any]
) -> neps.SearchSpace:
    """Return the neps space whose prior is the base config.

    Choices are JSON-encoded override mappings so grouped settings stay one
    categorical; the fidelity is the number of validation battles per model.
    """
    import neps

    confidence = tuning.prior_confidence
    parameters = {}
    for name, axis in tuning.search_space.items():
        if isinstance(axis, FloatRange):
            prior = _lookup(base, name)
            inside = (
                isinstance(prior, int | float) and axis.lower <= prior <= axis.upper
            )
            parameters[name] = neps.HPOFloat(
                lower=axis.lower,
                upper=axis.upper,
                log=axis.log,
                prior=prior if inside else None,
                prior_confidence=confidence,
            )
            continue
        overrides = [choice_overrides(name, value) for value in axis]
        choices = [json.dumps(choice, sort_keys=True) for choice in overrides]
        prior = next(
            (
                encoded
                for encoded, choice in zip(choices, overrides, strict=True)
                if all(_lookup(base, key) == value for key, value in choice.items())
            ),
            None,
        )
        parameters[name] = neps.HPOCategorical(
            choices=choices, prior=prior, prior_confidence=confidence
        )
    parameters[FIDELITY] = neps.HPOInteger(
        lower=tuning.min_battles_per_model,
        upper=tuning.max_battles_per_model,
        is_fidelity=True,
    )
    return neps.SearchSpace(parameters)


def decode_config(config: Mapping[str, Any]) -> tuple[dict[str, object], int]:
    """Return the dotted overrides and battles per model of a neps config."""
    overrides: dict[str, object] = {}
    for name, value in config.items():
        if name == FIDELITY:
            continue
        overrides.update(json.loads(value) if isinstance(value, str) else {name: value})
    return overrides, int(config[FIDELITY])


def apply_overrides(
    base: Mapping[str, object], overrides: Mapping[str, object]
) -> dict[str, object]:
    """Return a copy of ``base`` with dotted-path ``overrides`` assigned."""
    values = copy.deepcopy(dict(base))
    for path, value in overrides.items():
        *parents, leaf = path.split(".")
        node = values
        for key in parents:
            if node.get(key) is None:
                node[key] = {}
            node = node[key]
        node[leaf] = value
    return values
