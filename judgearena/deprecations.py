"""Translate deprecated experiment options before normal config validation."""

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model
from pydantic_settings import CliSettingsSource

from judgearena.log import get_logger

logger = get_logger(__name__)

# A simple rename needs one entry here for both YAML and the CLI.
RENAMES = {
    "elo.languages": "run.filters.lang",
    "generation.n_instructions": "run.sampling.limit",
    "elo.soft_elo_temperature": "protocol.judge.preference_temperature",
    "elo.calibrate_temperature": "protocol.judge.calibrate_temperature",
    "elo.calibration_size": "protocol.judge.calibration_size",
}

# These options change structure as well as name.
_RANDOM = "elo.elo_random_battles"
_QUOTA = "elo.n_instructions_per_language"
_METRIC_PARAMETERS = {
    "elo.n_bootstraps": "n_bootstraps",
    "elo.soft_elo": "soft",
    "elo.baseline_model": "baseline_model",
    "elo.fit_mode": "fit_mode",
}


class _LegacyElo(BaseModel):
    """Input types accepted by the previous CLI, including its list syntax."""

    model_config = ConfigDict(extra="forbid")
    baseline_model: str | None = None
    n_bootstraps: int = 20
    languages: list[str] | None = None
    n_instructions_per_language: int | None = None
    elo_random_battles: int | None = None
    soft_elo: bool = True
    soft_elo_temperature: float = 0.3
    calibrate_temperature: bool = False
    calibration_size: int | None = None
    fit_mode: Literal["joint", "frozen"] = "joint"


_ELO_KEYS = {f"elo.{name}" for name in _LegacyElo.model_fields}


class _LegacyGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    random_battles: int | None = None
    group_by: str | None = None
    per_group: int | None = None


def parse_cli_overrides(settings_cls, argv):
    """Parse historical flags with their original types before translating them."""
    args = []
    legacy_fields = {}
    for arg in argv:
        flag, sep, value = arg.partition("=")
        key = flag.removeprefix("--")
        if flag.startswith("--") and key in RENAMES and key not in _ELO_KEYS:
            target = RENAMES[key]
            logger.warning("%s is deprecated. Use --%s instead.", flag, target)
            arg = f"--{target}{sep}{value}"
        elif flag == "--elo" or flag.startswith("--elo."):
            legacy_fields["elo"] = (_LegacyElo | None, None)
        elif flag == "--protocol.graph" or flag.startswith("--protocol.graph."):
            arg = arg.replace("--protocol.graph", "--saved_protocol_graph", 1)
            legacy_fields["saved_protocol_graph"] = (_LegacyGraph | None, None)
        elif flag == "--graph" or flag.startswith("--graph."):
            legacy_fields["graph"] = (_LegacyGraph | None, None)
        args.append(arg)
    if legacy_fields:
        settings_cls = create_model(
            "LegacyCliConfig", __base__=settings_cls, **legacy_fields
        )
    values = CliSettingsSource(settings_cls, cli_parse_args=args)()
    protocol = values.get("protocol") or {}
    graph = values.pop("saved_protocol_graph", protocol.pop("graph", None))
    if graph is not None and values.get("graph") is None:
        values["graph"] = graph
    return values


def _section(owner, name):
    if owner.get(name) is None:
        owner[name] = {}
    return owner[name]


def _extract(values, key):
    """Remove a dotted key and prune sections emptied by its removal."""
    first, dot, rest = key.partition(".")
    if first not in values:
        return {}
    if not dot:
        return {key: values.pop(first)}
    section = values[first]
    if isinstance(section, dict) and (found := _extract(section, rest)):
        if not section:
            del values[first]
        return {key: found[rest]}
    return {}


def _translate_graph(values, inherited_protocol, instruction_limit):
    """Move the old graph route and its row-selection options to their owners."""
    protocol = values.get("protocol") or {}
    if "graph" not in values and "graph" not in protocol:
        return values
    task_graph = protocol.pop("graph", None)
    graph = values.pop("graph", None)
    if graph is None:
        graph = task_graph
    if graph is None:
        return values
    graph = _LegacyGraph.model_validate(graph).model_dump(exclude_none=True)
    protocol = _section(values, "protocol")
    if protocol.get("runner", inherited_protocol.get("runner")) == "meta_eval":
        raise ValueError(
            "Use meta_eval.top_models and meta_eval.battles_per_model to sample meta-evaluation battles."
        )
    baseline = protocol.get("baseline") or inherited_protocol.get("baseline") or {}
    if baseline.get("strategy") not in (None, "none", "recorded_opponents"):
        raise ValueError("graph conflicts with protocol.baseline.strategy")
    protocol["baseline"] = {"strategy": "recorded_opponents"}
    limit = graph.pop("random_battles", None)
    if limit is not None:
        if instruction_limit is not None:
            raise ValueError(
                "graph.random_battles conflicts with generation.n_instructions"
            )
        graph.update(method="random", limit=limit)
    if graph:
        sampling = _section(_section(values, "run"), "sampling")
        for key, value in graph.items():
            if (
                key in sampling
                and sampling[key] != value
                and str(sampling[key]) != str(value)
            ):
                raise ValueError(f"Conflicting graph and run.sampling.{key} settings")
            sampling[key] = value
    return values


def translate_deprecations(
    values: dict, inherited_protocol: dict, *, cli=False
) -> dict:
    protocol = values.get("protocol") or {}
    graph_declared = "graph" in values or "graph" in protocol
    if (
        not graph_declared
        and "elo" not in values
        and not any(
            rest in (values.get(section) or {})
            for section, rest in (key.split(".", 1) for key in RENAMES)
        )
    ):
        return values
    values = deepcopy(values)
    if graph_declared and (
        values.get("graph") is not None or protocol.get("graph") is not None
    ):
        logger.warning(
            "graph is deprecated. Use protocol.baseline.strategy and run.sampling instead."
        )
    options = {}
    # New aliases can share a section with historical typed options.
    for key in RENAMES:
        if key not in _ELO_KEYS:
            options.update(_extract(values, key))
    section = values.pop("elo", None)
    if section is not None:
        typed = _LegacyElo.model_validate(section).model_dump(exclude_unset=True)
        options.update({f"elo.{key}": value for key, value in typed.items()})
    prefix = "--" if cli else ""
    if options.get("elo.soft_elo") is False:
        judge = (values.get("protocol") or {}).get("judge") or {}
        if judge.get("preference_temperature") is not None or TypeAdapter(
            bool | None
        ).validate_python(judge.get("calibrate_temperature")):
            raise ValueError(
                f"Conflicting {prefix}elo.soft_elo and {prefix}protocol.judge settings"
            )

    explicit_metrics = "metrics" in (
        (values.get("protocol") or {}).get("scoring") or {}
    )
    for old, value in options.items():
        if old == "generation.n_instructions" and value is None:
            continue
        explicit = True
        if old in _METRIC_PARAMETERS:
            path = "protocol.scoring.metrics"
            scoring = _section(_section(values, "protocol"), "scoring")
            if "metrics" not in scoring:
                scoring["metrics"] = deepcopy(
                    inherited_protocol.get("scoring", {}).get("metrics", [])
                )
            candidates = [
                m for m in scoring["metrics"] if m["metric"] == "bradley_terry"
            ]
            targets = [
                m for m in candidates if m.get("name") in (None, "bradley_terry")
            ] or candidates
            if len(targets) != 1:
                raise ValueError(
                    f"{prefix}{old} needs one target metric. Use {prefix}{path} instead."
                )
            owner = targets[0].setdefault("parameters", {})
            changes = {_METRIC_PARAMETERS[old]: value}
            explicit = explicit_metrics
        elif old == _RANDOM:
            path = "run.sampling"
            owner = _section(values, "graph")
            changes = {"random_battles": value}
        elif old == _QUOTA:
            path = "run.sampling"
            owner = _section(values, "graph")
            changes = {
                "per_group": value,
                "group_by": "lang" if value is not None else None,
            }
        elif old in RENAMES:
            path = RENAMES[old]
            *parts, key = path.split(".")
            owner = values
            for part in parts:
                owner = _section(owner, part)
            changes = {key: value}
            if old == "elo.languages":
                if not value:
                    changes = {}
                elif key in owner:
                    from judgearena.datasets import resolve_dataset_filters

                    changes = {key: resolve_dataset_filters(None, owner, changes)[key]}
                    explicit = False
        else:
            raise ValueError(f"Unknown deprecated option {prefix}{old}")
        for field, replacement in changes.items():
            existing = owner.get(field)
            if (
                cli
                and old in _ELO_KEYS
                and explicit
                and field in owner
                and field != "group_by"
            ):
                annotation = _LegacyElo.model_fields[
                    old.removeprefix("elo.")
                ].annotation
                existing = TypeAdapter(annotation).validate_python(existing)
            if explicit and field in owner and existing != replacement:
                raise ValueError(
                    f"Conflicting {prefix}{old} and {prefix}{path} settings"
                )
            owner[field] = replacement
        if cli and path.startswith("run.filters."):
            path = "run.filters"
        logger.warning(
            "%s%s is deprecated. Use %s%s instead.", prefix, old, prefix, path
        )
    if options.get("elo.soft_elo") is False:
        judge = _section(_section(values, "protocol"), "judge")
        judge.update(preference_temperature=None, calibrate_temperature=False)
    return _translate_graph(
        values, inherited_protocol, options.get("generation.n_instructions")
    )
