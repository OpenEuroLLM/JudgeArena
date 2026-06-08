"""Hierarchical run configuration (pydantic-settings) with YAML loading."""

from __future__ import annotations
 
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings

from judgearena.cli_common import ELO_TASK_PREFIX, ELO_TASK_TO_ARENA
from judgearena.generate_and_evaluate import native_pairwise_baseline


class ModelArgs(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    path: str | None = None
    path_b: str | None = None
    max_out_tokens: int = 32768
    max_model_len: int | None = None
    chat_template: str | None = None
    engine_kwargs: dict = Field(default_factory=dict)


class JudgeArgs(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    max_out_tokens: int = 32768
    max_model_len: int | None = None
    engine_kwargs: dict = Field(default_factory=dict)
    provide_explanation: bool = False
    swap_mode: Literal["fixed", "both"] = "fixed"


class GenerationArgs(BaseModel):
    n_instructions: int | None = None
    truncate_all_input_chars: int = 8192
    truncate_judge_input_chars: int | None = None


class EloArgs(BaseModel):
    arena: str | None = None
    baseline_model: str | None = None
    n_bootstraps: int = 20
    languages: list[str] | None = None
    n_instructions_per_language: int | None = None


class RunArgs(BaseModel):
    seed: int = 0
    result_folder: str = "results"
    ignore_cache: bool = False
    use_tqdm: bool = False
    verbosity: int = 0
    log_file: str | None = None
    no_log_file: bool = False


class RunConfig(BaseSettings):
    model_config = ConfigDict(protected_namespaces=())

    task: str
    model: ModelArgs = Field(default_factory=ModelArgs)
    judge: JudgeArgs
    generation: GenerationArgs = Field(default_factory=GenerationArgs)
    elo: EloArgs | None = None
    run: RunArgs = Field(default_factory=RunArgs)

    @model_validator(mode="after")
    def _validate(self) -> RunConfig:
        is_elo = self.task.startswith(ELO_TASK_PREFIX)
        if is_elo:
            if self.elo is None:
                self.elo = EloArgs()
            if self.elo.arena is None:
                if self.task not in ELO_TASK_TO_ARENA:
                    raise ValueError(
                        f"Unknown elo task {self.task!r}; expected one of "
                        f"{list(ELO_TASK_TO_ARENA)}."
                    )
                self.elo.arena = ELO_TASK_TO_ARENA[self.task]
            if self.model.path is None:
                raise ValueError("model.path is required for elo tasks.")
            if self.model.path_b is not None:
                raise ValueError("model.path_b is not supported for elo tasks.")
        else:
            if self.elo is not None:
                raise ValueError("elo config is only valid for elo-* tasks.")
            if self.model.path is None:
                raise ValueError("model.path is required.")
            if (
                self.model.path_b is None
                and native_pairwise_baseline(self.task) is None
            ):
                raise ValueError(f"model.path_b is required for task {self.task!r}.")
        return self


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a RunConfig from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping.")
    return RunConfig(**data)
