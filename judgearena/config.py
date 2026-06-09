"""Hierarchical run configuration (pydantic-settings) with YAML loading."""

from __future__ import annotations
 
import argparse
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from judgearena.cli_common import ELO_TASK_PREFIX, ELO_TASK_TO_ARENA
from judgearena.generate_and_evaluate import native_pairwise_baseline

# Set by build_run_config() for the duration of RunConfig() construction.
_ACTIVE_CONFIG_PATH: str | None = None
_ACTIVE_CLI_ARGS: list[str] | None = None


class ModelArgs(BaseModel):
    """The model(s) under evaluation and their generation/engine settings."""

    model_config = ConfigDict(protected_namespaces=(), use_attribute_docstrings=True)

    name: str | None = None
    """Model under evaluation, formatted as ``{backend}/{model path}`` (e.g.
    ``VLLM/Qwen/Qwen2.5-0.5B-Instruct``). For elo tasks this is the single
    model rated against arena opponents."""

    baseline: str | None = None
    """Opponent model for pairwise tasks (the "Model B" reference). Omit for
    elo tasks; for pairwise tasks it defaults to the dataset-native baseline
    when left unset."""

    max_out_tokens: int = 32768
    """Generation token budget for each evaluated-model answer (for vLLM, keep
    this <= ``max_model_len``)."""

    max_model_len: int | None = None
    """Optional total context window (prompt + generation) for the generation
    vLLM instance. Applies to vLLM models only."""

    chat_template: str | None = None
    """Jinja2 chat template to use instead of the tokenizer's template (vLLM
    only; ignored by remote providers which template server-side)."""

    engine_kwargs: dict = Field(default_factory=dict)
    """JSON dict of engine-specific kwargs forwarded to the backend, e.g. for
    vLLM ``{"tensor_parallel_size": 2, "gpu_memory_utilization": 0.9}``."""


class JudgeArgs(BaseModel):
    """The judge model and how it scores each battle."""

    model_config = ConfigDict(protected_namespaces=(), use_attribute_docstrings=True)

    model: str
    """LLM used as the judge, in ``{backend}/{model path}`` format (e.g.
    ``OpenRouter/deepseek/deepseek-chat-v3.1``)."""

    max_out_tokens: int = 32768
    """Generation token budget for the judge response (reasoning + scores)."""

    max_model_len: int | None = None
    """Optional total context window for the judge vLLM instance."""

    engine_kwargs: dict = Field(default_factory=dict)
    """JSON dict of engine kwargs applied to the judge model only (overrides
    ``model.engine_kwargs`` for the judge)."""

    provide_explanation: bool = False
    """If set, the judge explains its reasoning before scoring. Aids
    interpretation; does not necessarily improve accuracy."""

    swap_mode: Literal["fixed", "both"] = "fixed"
    """Position-bias handling. ``fixed``: a single A-B judge pass. ``both``:
    judge each battle in both orders (A-B and B-A) and combine."""


class GenerationArgs(BaseModel):
    """How many instructions to use and input-length truncation."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    n_instructions: int | None = None
    """Number of instructions/battles to evaluate. Defaults to the full task."""

    truncate_all_input_chars: int = 8192
    """Character cap applied to each instruction before model generation."""

    truncate_judge_input_chars: int | None = None
    """Character cap applied to judge-side inputs before evaluation. Unset
    means no judge-side character truncation."""


class EloArgs(BaseModel):
    """Settings specific to elo-rating tasks (``--task elo-*``)."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    arena: str | None = None
    """Arena identifier whose battles supply the opponents. Derived from the
    ``elo-*`` task when left unset."""

    baseline_model: str | None = None
    """Model anchored at 1000 ELO; ratings are reported relative to it."""

    n_bootstraps: int = 20
    """Number of bootstrap resamples used for ELO confidence intervals."""

    languages: list[str] | None = None
    """Restrict arena battles to these language codes (e.g. ``["en", "fr"]``).
    Defaults to all languages."""

    n_instructions_per_language: int | None = None
    """Cap battles per language (useful for balanced multilingual eval)."""


class RunArgs(BaseModel):
    """Run-level settings: seed, output location, caching, and logging."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    seed: int = 0
    """Random seed for reproducibility."""

    result_folder: str = "results"
    """Directory where annotations, results, and the resolved ``config.yaml``
    are written (under a per-run subfolder)."""

    ignore_cache: bool = False
    """If set, ignore cached completions and regenerate them."""

    use_tqdm: bool = False
    """Show a tqdm progress bar (not compatible with vLLM)."""

    verbosity: int = 0
    """Logging verbosity (-1 quiet, 0 info, 1+ debug). Set on the CLI via
    ``-q`` / ``-v``."""

    log_file: str | None = None
    """Write the full DEBUG log to this file in addition to the console."""

    no_log_file: bool = False
    """Disable the automatic timestamped ``run-*.log`` in the result folder."""


class RunConfig(BaseSettings):
    model_config = SettingsConfigDict(
        protected_namespaces=(),
        nested_model_default_partial_update=True,
        cli_avoid_json=False,
        use_attribute_docstrings=True,
    )

    task: str
    """Benchmark to run. Generate+judge: ``alpaca-eval``, ``arena-hard-v2.0``,
    ``m-arena-hard-*``, ``mt-bench``, ``fluency-*``. ELO: ``elo-lmarena-100k``,
    ``elo-lmarena-140k``, ``elo-lmarena``, ``elo-comparia``."""

    model: ModelArgs = Field(default_factory=ModelArgs)
    """Model(s) under evaluation and their generation settings."""

    judge: JudgeArgs
    """The judge model and scoring behaviour."""

    generation: GenerationArgs = Field(default_factory=GenerationArgs)
    """Instruction count and input truncation."""

    elo: EloArgs | None = None
    """ELO-task settings (only for ``elo-*`` tasks)."""

    run: RunArgs = Field(default_factory=RunArgs)
    """Run-level settings (seed, output, caching, logging)."""

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
            if self.model.name is None:
                raise ValueError("model.name is required for elo tasks.")
            if self.model.baseline is not None:
                raise ValueError("model.baseline is not supported for elo tasks.")
        else:
            if self.elo is not None:
                raise ValueError("elo config is only valid for elo-* tasks.")
            if self.model.name is None:
                raise ValueError("model.name is required.")
            if (
                self.model.baseline is None
                and native_pairwise_baseline(self.task) is None
            ):
                raise ValueError(f"model.baseline is required for task {self.task!r}.")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Precedence: CLI flags first (highest), then --config_path YAML, then defaults.
        # When neither global is set (direct construction / load_config), use init kwargs.
        sources: list[PydanticBaseSettingsSource] = []
        if _ACTIVE_CLI_ARGS is not None:
            sources.append(
                CliSettingsSource(settings_cls, cli_parse_args=_ACTIVE_CLI_ARGS)
            )
        if _ACTIVE_CONFIG_PATH is not None:
            sources.append(
                YamlConfigSettingsSource(settings_cls, yaml_file=_ACTIVE_CONFIG_PATH)
            )
        return tuple(sources) or (init_settings,)


def build_run_config(argv: list[str] | None = None) -> RunConfig:
    """Build a RunConfig from CLI flags and an optional --config_path YAML.

    Precedence: CLI flags > --config_path YAML > model defaults.
    """
    global _ACTIVE_CONFIG_PATH, _ACTIVE_CLI_ARGS
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config_path", default=None)
    pre.add_argument("-v", "--verbose", action="count", default=0)
    pre.add_argument("-q", "--quiet", action="store_true")
    pre_args, rest = pre.parse_known_args(argv)

    _ACTIVE_CONFIG_PATH = pre_args.config_path
    _ACTIVE_CLI_ARGS = rest
    try:
        cfg = RunConfig()
    finally:
        _ACTIVE_CONFIG_PATH = None
        _ACTIVE_CLI_ARGS = None

    cfg.run.verbosity = -1 if pre_args.quiet else pre_args.verbose
    return cfg


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a RunConfig from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping.")
    return RunConfig(**data)


def dump_config(cfg: RunConfig, path: str | Path) -> None:
    """Write the resolved config as YAML (round-trippable via ``--config_path``)."""
    Path(path).write_text(
        yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8"
    )
