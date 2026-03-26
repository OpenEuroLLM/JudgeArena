"""Schema for multi-criteria evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        for score in sorted(criterion.score_references.keys(), reverse=True)
    )
    return f"{base}\n   Score references:\n{refs}"


def prompt_block(criteria: list[Criterion]) -> str:
    """Render criteria as scoring instructions."""
    lines = ["Score the following completion on each criterion:\n"]
    for i, criterion in enumerate(criteria, 1):
        lines.append(f"{i}. {criterion_prompt_block(criterion)}")
    return "\n".join(lines)


def criterion_names(criteria: list[Criterion]) -> list[str]:
    """Return criterion names in order."""
    return [criterion.name for criterion in criteria]


@dataclass
class CriteriaScore:
    """Scores for a single (instruction, completion) pair across all criteria."""

    instruction_index: int | str
    model: str
    scores: dict[str, float]
    raw_judge_output: str | None = None

    def to_list(self, criterion_names: list[str]) -> list[float]:
        """Return scores as an ordered vector matching criterion_names."""
        return [self.scores.get(name, 0.0) for name in criterion_names]


def criterion_from_dict(data: dict[str, Any]) -> Criterion:
    """Build a Criterion from a serialized mapping."""
    return Criterion(
        name=data["name"],
        description=data["description"],
        score_references=data.get("score_references", {}),
    )


def criteria_from_dict(data: dict[str, Any]) -> list[Criterion]:
    """Build criteria from a serialized mapping."""
    return [criterion_from_dict(item) for item in data["criteria"]]
