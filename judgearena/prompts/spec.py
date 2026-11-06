"""Prompt selection shared by task definitions and run configurations."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from judgearena.prompts.parsing import parser_file_reference, resolve_judge_parser
from judgearena.utils.extensions import config_definitions


def prompt_file_fields(prompt):
    """Yield owners and keys of template and preset-option files."""
    option_files = ()
    if prompt.get("preset") is not None:
        from judgearena.prompts.registry import prompt_option_files

        option_files = prompt_option_files(prompt["preset"])

    for owner, names in (
        (prompt, ("file", "system_file", "user_file")),
        (prompt.get("options"), option_files),
    ):
        if isinstance(owner, dict):
            for name in names:
                if owner.get(name) is not None:
                    yield owner, name


def resolve_prompt_paths(prompt, directory):
    reference = parser_file_reference(prompt.get("parser"))
    if reference is not None:
        path, symbol = reference
        if not path.is_absolute():
            prompt["parser"] = f"{directory / path}:{symbol}"
    for owner, name in prompt_file_fields(prompt):
        value = owner[name]
        if isinstance(value, (str, Path)) and not Path(value).is_absolute():
            owner[name] = str(directory / value)


def config_prompts(data):
    """Yield prompt settings in a run or task, including category overrides."""
    judges = [data.get("judge")]
    for definition in config_definitions(data):
        protocol = definition.get("protocol")
        if isinstance(protocol, dict):
            judges.append(protocol.get("judge"))
    for judge in judges:
        if isinstance(judge, dict):
            categories = judge.get("category_prompts")
            selections = categories.values() if isinstance(categories, dict) else ()
            for prompt in (judge.get("prompt"), *selections):
                if isinstance(prompt, dict):
                    yield prompt


class JudgePromptSpec(BaseModel):
    """Select a preset or custom template/parser, optionally from a YAML bundle."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    file: Path | None = Field(default=None, exclude=True)
    """YAML bundle expanded into effective prompt settings when loaded."""

    user: str | None = None
    """Inline user template, paired with a named parser."""
    system: str | None = None
    """Optional inline system template."""

    preset: str | None = None
    """Named prompt/parser bundle."""

    system_file: Path | None = None
    """Path to the judge system prompt file."""

    user_file: Path | None = None
    """Path to the judge user-prompt template file."""

    parser: str | None = None
    """Parser for custom files; defaults to ``score``. Presets own their parser."""

    options: dict[str, object] = Field(default_factory=dict)
    """Settings accepted by the selected preset's configuration function."""

    @model_validator(mode="before")
    @classmethod
    def _load_bundle(cls, values):
        if not isinstance(values, dict) or values.get("file") is None:
            return values
        if any(
            value is not None and value != {}
            for key, value in values.items()
            if key != "file"
        ):
            raise ValueError("A prompt file owns its templates, parser and options.")
        path = Path(values["file"])
        with path.open(encoding="utf-8") as stream:
            bundle = yaml.safe_load(stream)
        if not isinstance(bundle, dict) or "file" in bundle:
            raise ValueError(
                "A prompt bundle must be a mapping without nested file references."
            )
        resolve_prompt_paths(bundle, path.resolve().parent)
        return bundle

    @model_validator(mode="after")
    def _validate_source(self) -> JudgePromptSpec:
        if (
            sum(value is not None for value in (self.preset, self.user_file, self.user))
            != 1
        ):
            raise ValueError(
                "Select one of prompt.preset, prompt.user_file or prompt.user."
            )
        if self.system is not None and self.system_file is not None:
            raise ValueError("Select either prompt.system or prompt.system_file.")
        if self.preset is not None and any(
            value is not None for value in (self.system_file, self.system, self.parser)
        ):
            raise ValueError("A prompt preset owns its templates and parser.")
        if self.options and self.preset is None:
            raise ValueError("Prompt options require a preset.")
        if self.user_file is not None or self.user is not None:
            resolve_judge_parser(self.parser or "score")
        return self
