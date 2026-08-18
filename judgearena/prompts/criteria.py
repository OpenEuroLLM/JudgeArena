"""Criteria definitions, loading, validation, and prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

SCALE_MIN = 1
SCALE_MAX = 10


@dataclass
class Criterion:
    """A single scoring criterion."""

    name: str
    description: str
    score_references: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[int, str] = {}
        for raw_score, text in self.score_references.items():
            score = int(raw_score)
            if not (SCALE_MIN <= score <= SCALE_MAX):
                raise ValueError(
                    f"Score reference {score} for '{self.name}' is outside "
                    f"the configured scale [{SCALE_MIN}, {SCALE_MAX}]"
                )
            normalized[score] = str(text).strip()
        self.score_references = normalized


def criterion_prompt_block(criterion: Criterion) -> str:
    """Render a criterion as a scoring instruction for the judge."""
    base = (
        f"**{criterion.name.title()}** ({SCALE_MIN}–{SCALE_MAX}): "
        f"{criterion.description}"
    )
    if not criterion.score_references:
        return base

    refs = "\n".join(
        f"   - {score}: {criterion.score_references[score]}"
        for score in sorted(criterion.score_references, reverse=True)
    )
    return f"{base}\n   Score references:\n{refs}"


def prompt_block(criteria: list[Criterion]) -> str:
    """Render criteria as scoring instructions."""
    lines = ["Score the following completion on each criterion:\n"]
    for index, criterion in enumerate(criteria, 1):
        lines.append(f"{index}. {criterion_prompt_block(criterion)}")
    return "\n".join(lines)


def criterion_names(criteria: list[Criterion]) -> list[str]:
    """Return criterion names in order."""
    return [criterion.name for criterion in criteria]


def criterion_from_dict(data: dict[str, Any]) -> Criterion:
    """Build a criterion from a serialized mapping."""
    return Criterion(
        name=data["name"],
        description=data["description"],
        score_references=data.get("score_references", {}),
    )


def criteria_from_dict(data: dict[str, Any]) -> list[Criterion]:
    """Build criteria from a serialized mapping."""
    return [criterion_from_dict(item) for item in data["criteria"]]


def _load_criteria_data(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported criteria file format '{path.suffix}'. Use .yaml or .yml."
        )

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Criteria YAML must define a mapping at the top level.")
    return data


def load_criteria_from_file(path: str | Path) -> list[Criterion]:
    """Load criteria from a YAML file path."""
    return criteria_from_dict(_load_criteria_data(path))


def _load_builtin_criteria(filename: str) -> list[Criterion]:
    text = (
        files("judgearena.prompts")
        .joinpath("templates")
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Built-in criteria file '{filename}' must define a mapping.")
    return criteria_from_dict(data)


CRITERIA_BY_NAME: dict[str, list[Criterion]] = {
    "default": _load_builtin_criteria("criteria-default.yaml"),
}


def resolve_criteria(
    criteria_name: str = "default",
    criteria_file: str | Path | None = None,
) -> tuple[str, list[Criterion]]:
    """Resolve criteria by name or file path, with a file taking precedence."""
    if criteria_file is not None:
        path = Path(criteria_file)
        data = _load_criteria_data(path)
        resolved_name = data.get("name", path.stem)
        return resolved_name, criteria_from_dict(data)
    if criteria_name not in CRITERIA_BY_NAME:
        available = ", ".join(sorted(CRITERIA_BY_NAME))
        raise KeyError(f"Unknown criteria '{criteria_name}'. Available: {available}")
    return criteria_name, CRITERIA_BY_NAME[criteria_name]
