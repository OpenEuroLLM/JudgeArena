"""Construct NePS domains using its public constructor arguments."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import neps

FIDELITY = "meta_eval.battles_per_model"
DOMAIN_TYPES = frozenset({"Categorical", "Float", "Integer", "IntegerFidelity"})


def parameter_specs(search_space: Mapping, base: Mapping, fidelity: Mapping) -> dict:
    """Flatten nested parameter definitions and resolve value-based priors."""
    specs = {}

    def visit(section, prefix=""):
        for key, value in section.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, Mapping) and not (
                {"choices", "lower", "upper"} & value.keys()
            ):
                visit(value, name)
                continue
            if not (
                name.startswith("judge.")
                or name == "generation.truncate_judge_input_chars"
            ):
                raise ValueError(
                    f"Search space must not override tuner-owned keys: {name}"
                )
            spec = {"choices": value} if isinstance(value, list) else dict(value)
            node = base
            for part in name.split("."):
                node = node.get(part) if isinstance(node, Mapping) else None
            explicit_prior = "prior" in spec
            prior = spec.pop("prior", node)
            confidence = spec.pop("prior_confidence", "medium")
            if "choices" in spec:
                kind = "Categorical"
                if explicit_prior and prior not in spec["choices"]:
                    raise ValueError(f"Prior for {name} is not in its choices")
                prior = (
                    spec["choices"].index(prior) if prior in spec["choices"] else None
                )
            else:
                kind = (
                    "Integer"
                    if all(type(spec[k]) is int for k in ("lower", "upper"))
                    else "Float"
                )
                if prior is not None and not spec["lower"] <= prior <= spec["upper"]:
                    if explicit_prior:
                        raise ValueError(f"Prior for {name} is outside its range")
                    prior = None
            if prior is not None:
                spec.update(prior=prior, prior_confidence=confidence)
            specs[name] = {"type": kind, **spec}

    visit(search_space)
    specs[FIDELITY] = {"type": "IntegerFidelity", **fidelity["battles_per_model"]}
    return specs


def build_neps_space(specs: Mapping[str, dict]) -> neps.PipelineSpace:
    """Translate constructor tags to domains without changing their arguments."""
    import neps

    parameters = {}
    for name, spec in specs.items():
        kind = spec["type"]
        if kind not in DOMAIN_TYPES:
            raise ValueError(f"Unsupported NePS domain: {kind}")
        if name == FIDELITY:
            if kind != "IntegerFidelity":
                raise ValueError(f"{FIDELITY} must use IntegerFidelity")
        elif not (
            name.startswith("judge.") or name == "generation.truncate_judge_input_chars"
        ):
            raise ValueError(f"Search space must not override tuner-owned keys: {name}")
        elif kind == "IntegerFidelity":
            raise ValueError(f"Only {FIDELITY} may be a fidelity")
        kwargs = {key: value for key, value in spec.items() if key != "type"}
        if kind == "Categorical":
            kwargs["choices"] = tuple(kwargs["choices"])
        parameters[name] = getattr(neps, kind)(**kwargs)
    if FIDELITY not in parameters:
        raise ValueError(f"Search space requires {FIDELITY}")
    return type("JudgeSpace", (neps.PipelineSpace,), parameters)()


def axis_overrides(specs: Mapping[str, dict]) -> Iterator[dict[str, object]]:
    """Yield candidate values for preflight RunConfig validation."""
    for name, spec in specs.items():
        if name == FIDELITY:
            continue
        values = (
            spec["choices"]
            if spec["type"] == "Categorical"
            else [spec["lower"], spec["upper"]]
        )
        for value in values:
            yield {name: value}


def decode_config(config: Mapping[str, Any]) -> tuple[dict[str, object], int]:
    """Split the sampled judge settings from the evaluation fidelity."""
    overrides = dict(config)
    return overrides, int(overrides.pop(FIDELITY))


def apply_overrides(
    base: Mapping[str, object], overrides: Mapping[str, object]
) -> dict[str, object]:
    """Return a copy of base with dotted-path overrides assigned."""
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
