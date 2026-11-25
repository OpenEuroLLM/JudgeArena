"""Prompt selection shared by task definitions and run configurations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from judgearena.prompts.parsing import (
    ParsedPreference,
    parser_file_reference,
    resolve_judge_parser,
)
from judgearena.utils.extensions import config_definitions, python_object_reference


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


def config_prompts(data, *, expand_names=False):
    """Yield prompt settings in a run or task, including category overrides."""
    if expand_names:
        from judgearena.prompts.registry import PRESETS

    judges = [data.get("judge")]
    for definition in config_definitions(data):
        protocol = definition.get("protocol")
        if isinstance(protocol, dict):
            judges.append(protocol.get("judge"))
    for judge in judges:
        if isinstance(judge, dict):
            if (
                expand_names
                and judge.get("prompt") is None
                and (preset := PRESETS.get(judge.get("default_prompt_preset")))
                is not None
                and preset.bundle_spec is not None
            ):
                judge["prompt"] = {"preset": judge["default_prompt_preset"]}
            categories = judge.get("category_prompts")
            if expand_names and isinstance(categories, dict):
                for category, selection in categories.items():
                    if (
                        isinstance(selection, str)
                        and (preset := PRESETS.get(selection)) is not None
                        and preset.bundle_spec is not None
                    ):
                        categories[category] = {"preset": selection}
            selections = categories.values() if isinstance(categories, dict) else ()
            for prompt in (judge.get("prompt"), *selections):
                if isinstance(prompt, dict):
                    yield prompt


def expand_registered_prompt(prompt):
    """Pin a registered YAML bundle's settings before saving a run."""
    from judgearena.prompts.registry import PRESETS

    name = prompt.get("name") or prompt.get("preset")
    while (preset := PRESETS.get(prompt.get("preset"))) is not None:
        if preset.bundle_spec is None:
            break
        expanded = deepcopy(preset.bundle_spec)
        expanded["options"] = {
            **expanded.get("options", {}),
            **prompt.get("options", {}),
        }
        expanded["name"] = name
        prompt.clear()
        prompt.update(expanded)


def serialize_run_prompts(cfg, saved):
    """Pin prompt settings in a saved config while preserving category routing."""
    runtime_prompt = cfg.judge.prompt
    task_judge = cfg.resolved_task.spec.protocol.judge
    if (
        runtime_prompt is not None
        and runtime_prompt.preset is not None
        and not runtime_prompt.options
        and (
            runtime_prompt == task_judge.prompt
            or (
                task_judge.prompt is None
                and runtime_prompt.preset == task_judge.default_prompt_preset
            )
        )
    ):
        # Inherit the identical task selection, including its category routing.
        saved["judge"]["prompt"] = None
    for prompt in config_prompts(saved, expand_names=True):
        expand_registered_prompt(prompt)
        parser = (
            resolve_judge_parser(prompt["parser"]) if prompt.get("parser") else None
        )
        if parser is not None and (reference := python_object_reference(parser)):
            prompt["parser"] = reference


class JudgePromptSpec(BaseModel):
    """Select a preset or custom template/parser, optionally from a YAML bundle."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    name: str | None = None
    """Optional display name, retained when a registered bundle is saved for replay."""

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

    judging_mode: Literal["pairwise", "single_answer"] | None = None
    """Candidate inputs for a standalone prompt; presets supply their own mode."""

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
        if self.preset is not None and self.judging_mode is not None:
            raise ValueError("A prompt preset owns its judging mode.")
        if self.options and self.preset is None:
            raise ValueError("Prompt options require a preset.")
        if self.user_file is not None or self.user is not None:
            parser = resolve_judge_parser(self.parser or "score")
            if self.judging_mode is None:
                # Preserve custom configs written before bundles declared their mode.
                self.judging_mode = (
                    "pairwise"
                    if issubclass(parser.result_type, ParsedPreference)
                    else "single_answer"
                )
        return self
