"""Load and render scoring rubrics for the criteria prompt preset."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from judgearena.prompts.registry import ResolvedJudgePrompt

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
    lines = ["Score each answer on every criterion:\n"]
    for i, criterion in enumerate(criteria, 1):
        lines.append(f"{i}. {criterion_prompt_block(criterion)}")
    return "\n".join(lines)


def criteria_from_dict(data: dict[str, Any]) -> list[Criterion]:
    """Build criteria from a serialized mapping."""
    return [
        Criterion(
            name=item["name"],
            description=item["description"],
            score_references=item.get("score_references", {}),
        )
        for item in data["criteria"]
    ]


def _load_builtin_criteria(filename: str) -> list[Criterion]:
    data = yaml.safe_load(
        files("judgearena.prompts").joinpath("templates").joinpath(filename).read_text()
    )
    if not isinstance(data, dict):
        raise ValueError(f"Built-in criteria file '{filename}' must define a mapping.")
    return criteria_from_dict(data)


def _load_criteria_data(path: str | Path) -> dict:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported criteria file format '{path.suffix}'. Use .yaml or .yml."
        )

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Criteria YAML must define a mapping at the top level.")
    return data


def load_criteria_from_file(path: str | Path) -> list[Criterion]:
    """Load criteria from a YAML file path."""
    return criteria_from_dict(_load_criteria_data(path))


def resolve_criteria(criteria_file: str | Path | None = None) -> list[Criterion]:
    """Load a custom rubric or the criteria presets' packaged default."""
    if criteria_file is not None:
        return load_criteria_from_file(criteria_file)
    return _load_builtin_criteria("criteria-default.yaml")


def configure_criteria_prompt(
    prompt: ResolvedJudgePrompt, *, criteria_file: str | Path | None = None
) -> ResolvedJudgePrompt:
    """Build matching instructions and a parser from the selected rubric."""
    from judgearena.prompts.parsing import CriteriaParser, CriteriaScoreParser

    criteria = resolve_criteria(criteria_file)
    names = [criterion.name for criterion in criteria]
    pairwise = prompt.judging_mode == "pairwise"
    parser = CriteriaParser(names) if pairwise else CriteriaScoreParser(names)
    fields = {
        "criteria_block": prompt_block(criteria),
        "criteria_output_lines": "\n".join(
            f"{name}: A=<score>, B=<score>" for name in names
        )
        if pairwise
        else json.dumps(
            {"criteria": dict.fromkeys(names, "<score>")}, ensure_ascii=False, indent=2
        ),
    }

    def render(template: str) -> str:
        # Substitute once: rubric text may contain literal braces or placeholder names.
        return re.sub(
            r"\{(criteria_block|criteria_output_lines)\}",
            lambda match: fields[match[1]].replace("{", "{{").replace("}", "}}"),
            template,
        )

    return replace(
        prompt,
        parser=parser,
        user_prompt_template=render(prompt.user_prompt_template),
        output_format=render(prompt.output_format)
        if prompt.output_format is not None
        else None,
    )
