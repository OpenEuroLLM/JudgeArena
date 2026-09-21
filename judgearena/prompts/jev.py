"""Packaged prompt presets for the Jev System One backend."""

from dataclasses import dataclass
from importlib.resources import files

import yaml


@dataclass(frozen=True)
class JevPromptPreset:
    name: str
    parser: str
    decision_mode: str
    task_kind: str
    system_prompt: str
    user_prompt_template: str
    questions: dict[str, dict]


def _load_preset(filename: str) -> JevPromptPreset:
    data = yaml.safe_load(
        files("judgearena.prompts")
        .joinpath("data", "jev", filename)
        .read_text(encoding="utf-8")
    )
    if not isinstance(data, dict) or not isinstance(data.get("questions"), dict):
        raise ValueError(f"Jev prompt file {filename!r} must define a questions map.")
    try:
        return JevPromptPreset(
            name=data["name"],
            parser=data["parser"],
            decision_mode=data["decision_mode"],
            task_kind=data["task_kind"],
            system_prompt=data["system_prompt"],
            user_prompt_template=data["user_prompt_template"],
            questions=data["questions"],
        )
    except KeyError as error:
        raise ValueError(
            f"Jev prompt file {filename!r} is missing {error.args[0]!r}."
        ) from error


JEV_PROMPT_PRESETS = {
    preset.name: preset
    for preset in (
        _load_preset("typesafe-choice.yaml"),
        _load_preset("typesafe-comparative-score.yaml"),
        _load_preset("typesafe-pair-score.yaml"),
        _load_preset("typesafe-fluency-choice.yaml"),
    )
}

JEV_QUESTION_MODES: dict[str, dict[str, dict]] = {}
for preset in JEV_PROMPT_PRESETS.values():
    existing = JEV_QUESTION_MODES.setdefault(preset.decision_mode, preset.questions)
    if existing != preset.questions:
        raise ValueError(
            f"Jev presets using {preset.decision_mode!r} must define the same questions."
        )
