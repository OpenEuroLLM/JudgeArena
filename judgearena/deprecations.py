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
    "elo.elo_random_battles": "graph.random_battles",
    "elo.soft_elo_temperature": "protocol.judge.preference_temperature",
    "elo.calibrate_temperature": "protocol.judge.calibrate_temperature",
    "elo.calibration_size": "protocol.judge.calibration_size",
}

# These options change structure as well as name.
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


def parse_cli_overrides(settings_cls, argv):
    """Use the ordinary settings parser for all released Elo flags."""
    args = []
    legacy_elo = False
    for arg in argv:
        flag, sep, value = arg.partition("=")
        key = flag.removeprefix("--")
        if flag.startswith("--") and key in RENAMES and key not in _ELO_KEYS:
            target = RENAMES[key]
            logger.warning("%s is deprecated. Use --%s instead.", flag, target)
            arg = f"--{target}{sep}{value}"
        elif flag == "--elo" or flag.startswith("--elo."):
            legacy_elo = True
        args.append(arg)
    if legacy_elo:
        settings_cls = create_model(
            "LegacyCliConfig", __base__=settings_cls, elo=(_LegacyElo | None, None)
        )
    return CliSettingsSource(settings_cls, cli_parse_args=args)()


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


def translate_deprecations(
    values: dict, inherited_protocol: dict, *, cli=False
) -> dict:
    keys = RENAMES.keys() | _METRIC_PARAMETERS.keys() | {_QUOTA}
    if not any(k.partition(".")[0] in values for k in keys):
        return values
    values = deepcopy(values)
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
        elif old == _QUOTA:
            path = "graph"
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
    return values
