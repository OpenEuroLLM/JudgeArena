"""Criteria-set I/O helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from judgearena.criteria.defaults import CRITERIA_BY_NAME
from judgearena.criteria.schema import Criterion, criteria_from_dict


def _load_criteria_data(path: str | Path) -> dict:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported criteria file format '{path.suffix}'. "
            "Use .yaml or .yml."
        )

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Criteria YAML must define a mapping at the top level.")
    return data


def load_criteria_from_file(path: str | Path) -> list[Criterion]:
    """Load criteria from a YAML file path."""
    return criteria_from_dict(_load_criteria_data(path))


def resolve_criteria(
    criteria_name: str = "default",
    criteria_file: str | Path | None = None,
) -> tuple[str, list[Criterion]]:
    """Resolve criteria by name or file path (file path takes precedence)."""
    if criteria_file is not None:
        path = Path(criteria_file)
        data = _load_criteria_data(path)
        resolved_name = data.get("name", path.stem)
        return resolved_name, criteria_from_dict(data)
    if criteria_name not in CRITERIA_BY_NAME:
        available = ", ".join(sorted(CRITERIA_BY_NAME.keys()))
        raise KeyError(f"Unknown criteria '{criteria_name}'. Available: {available}")
    return criteria_name, CRITERIA_BY_NAME[criteria_name]
