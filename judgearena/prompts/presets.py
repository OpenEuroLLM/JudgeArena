"""Packaged judge prompt preset definitions."""

from dataclasses import dataclass
from importlib.resources import files

import yaml


@dataclass(frozen=True)
class PromptPresetSpec:
    name: str
    source_path: str
    parser: str | None
    system_prompt: str | None
    user_prompt_template: str | None
    delegated: bool = False
    with_explanation: bool = False
    decision_mode: str | None = None
    task_kind: str | None = None
    questions: dict[str, dict] | None = None
    criteria_scoring: dict | None = None


def _load_preset(filename: str) -> PromptPresetSpec:
    data = yaml.safe_load(
        files("judgearena.prompts")
        .joinpath("data", "presets", filename)
        .read_text(encoding="utf-8")
    )
    if not isinstance(data, dict):
        raise ValueError(f"Prompt preset file {filename!r} must define a map.")

    name = data.get("name")
    if name != filename.removesuffix(".yaml"):
        raise ValueError(f"Prompt preset file {filename!r} has invalid name {name!r}.")
    delegated = data.get("delegated", False)
    parser = data.get("parser")
    system_prompt = data.get("system_prompt")
    user_prompt_template = data.get("user_prompt_template")
    if delegated:
        if any(
            value is not None for value in (parser, system_prompt, user_prompt_template)
        ):
            raise ValueError(f"Delegated prompt preset {name!r} cannot define prompts.")
    elif not all(
        isinstance(value, str)
        for value in (parser, system_prompt, user_prompt_template)
    ):
        raise ValueError(
            f"Prompt preset {name!r} must define parser, system_prompt, and "
            "user_prompt_template."
        )

    decision_mode = data.get("decision_mode")
    task_kind = data.get("task_kind")
    questions = data.get("questions")
    jev_fields = (decision_mode, task_kind, questions)
    if any(value is not None for value in jev_fields) and not (
        isinstance(decision_mode, str)
        and isinstance(task_kind, str)
        and isinstance(questions, dict)
    ):
        raise ValueError(
            f"Jev prompt preset {name!r} must define decision_mode, task_kind, "
            "and questions."
        )

    return PromptPresetSpec(
        name=name,
        source_path=f"data/presets/{filename}",
        parser=parser,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        delegated=delegated,
        with_explanation=data.get("with_explanation", False),
        decision_mode=decision_mode,
        task_kind=task_kind,
        questions=questions,
        criteria_scoring=data.get("criteria_scoring"),
    )


_PRESET_FILES = (
    "default.yaml",
    "typesafe-choice.yaml",
    "typesafe-comparative-score.yaml",
    "typesafe-pair-score.yaml",
    "typesafe-criteria-score.yaml",
    "typesafe-criteria-choice.yaml",
    "typesafe-criteria-comparative-score.yaml",
    "typesafe-fluency-choice.yaml",
    "default_with_explanation.yaml",
    "fluency.yaml",
    "fastchat-pairwise.yaml",
    "arena-hard.yaml",
    "arena-hard-creative.yaml",
    "alpaca-eval.yaml",
    "meta-eval-pair-score.yaml",
    "meta-eval-alpaca-eval-json.yaml",
    "meta-eval-alpaca-eval-pair-score.yaml",
)

PROMPT_PRESETS = {preset.name: preset for preset in map(_load_preset, _PRESET_FILES)}
