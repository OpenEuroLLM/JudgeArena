"""Built-in criteria definitions."""

from __future__ import annotations

from importlib.resources import files

import yaml

from judgearena.criteria.schema import Criterion, criteria_from_dict


def _load_builtin_criteria(filename: str) -> list[Criterion]:
    data = yaml.safe_load(
        files("judgearena.criteria").joinpath("data").joinpath(filename).read_text()
    )
    if not isinstance(data, dict):
        raise ValueError(f"Built-in criteria file '{filename}' must define a mapping.")
    return criteria_from_dict(data)


CRITERIA_BY_NAME: dict[str, list[Criterion]] = {
    "default": _load_builtin_criteria("default.yaml"),
}
