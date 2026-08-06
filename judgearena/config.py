"""Hierarchical run configuration (pydantic-settings) with YAML loading."""

from __future__ import annotations

import argparse
import getpass
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    CliImplicitFlag,
    CliSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from judgearena.constants import ELO_TASK_PREFIX, ELO_TASK_TO_ARENA, META_EVAL_TASK
from judgearena.inference_cache import InferenceCache
from judgearena.pairwise_baselines import native_pairwise_baseline
from judgearena.store_sync import DEFAULT_CACHE_REPO

CacheMode = Literal["use", "off", "refresh"]

_CACHE_CLI_SHORTCUTS = {
    "cache.store_root": "store_root",
    "cache.cache_mode": "cache_mode",
    "cache.cache_hf_repo": "cache_hf_repo",
    "cache.cache_fetch": "cache_fetch",
    "cache.cache_push": "cache_push",
    "cache.cache_create_pr": "cache_create_pr",
    "cache.pushed_by": "pushed_by",
}

# Set by build_run_config() for the duration of RunConfig() construction.
_ACTIVE_CONFIG_PATH: str | None = None
_ACTIVE_CLI_ARGS: list[str] | None = None


def _drop_none(kwargs: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in kwargs.items() if v is not None}


def default_pushed_by() -> str:
    return getpass.getuser()


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

    temperature: float | None = None
    """Sampling temperature for the evaluated model. Unset keeps the backend
    default."""

    top_p: float | None = None
    """Nucleus-sampling probability for the evaluated model. Unset keeps the
    backend default."""

    top_k: int | None = None
    """Top-k sampling cutoff for the evaluated model when supported."""

    seed: int | None = None
    """Backend sampling seed for the evaluated model when supported."""

    max_model_len: int | None = None
    """Optional total context window (prompt + generation) for the generation
    vLLM instance. Applies to vLLM models only."""

    chat_template: str | None = None
    """Jinja2 chat template to use instead of the tokenizer's template (vLLM
    only; ignored by remote providers which template server-side)."""

    engine_kwargs: dict = Field(default_factory=dict)
    """JSON dict of engine-specific kwargs forwarded to the backend, e.g. for
    vLLM ``{"tensor_parallel_size": 2, "gpu_memory_utilization": 0.9}``."""

    baseline_max_out_tokens: int | None = None
    """Generation token budget for the baseline/opponent model. Unset inherits
    ``model.max_out_tokens``."""

    baseline_temperature: float | None = None
    """Sampling temperature for the baseline/opponent model. Unset inherits
    ``model.temperature``."""

    baseline_top_p: float | None = None
    """Nucleus-sampling probability for the baseline/opponent model. Unset
    inherits ``model.top_p``."""

    baseline_top_k: int | None = None
    """Top-k sampling cutoff for the baseline/opponent model when supported.
    Unset inherits ``model.top_k``."""

    baseline_seed: int | None = None
    """Backend sampling seed for the baseline/opponent model when supported.
    Unset inherits ``model.seed``."""

    baseline_max_model_len: int | None = None
    """Optional total context window for the baseline/opponent vLLM instance.
    Unset inherits ``model.max_model_len``."""

    baseline_chat_template: str | None = None
    """Jinja2 chat template for the baseline/opponent model. Unset inherits
    ``model.chat_template``."""

    baseline_engine_kwargs: dict | None = None
    """JSON dict of engine-specific kwargs for the baseline/opponent model.
    Unset inherits ``model.engine_kwargs``."""

    def evaluated_generation_kwargs(self) -> dict[str, object]:
        """Kwargs for model A / the evaluated model."""
        kwargs: dict[str, object] = dict(self.engine_kwargs)
        kwargs.update(
            _drop_none(
                {
                    "max_tokens": self.max_out_tokens,
                    "max_model_len": self.max_model_len,
                    "chat_template": self.chat_template,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "seed": self.seed,
                }
            )
        )
        return kwargs

    def baseline_generation_kwargs(self) -> dict[str, object]:
        """Kwargs for model B / the baseline model."""
        engine_kwargs = (
            self.engine_kwargs
            if self.baseline_engine_kwargs is None
            else self.baseline_engine_kwargs
        )
        kwargs: dict[str, object] = dict(engine_kwargs)
        kwargs.update(
            _drop_none(
                {
                    "max_tokens": (
                        self.baseline_max_out_tokens
                        if self.baseline_max_out_tokens is not None
                        else self.max_out_tokens
                    ),
                    "max_model_len": (
                        self.baseline_max_model_len
                        if self.baseline_max_model_len is not None
                        else self.max_model_len
                    ),
                    "chat_template": (
                        self.baseline_chat_template
                        if self.baseline_chat_template is not None
                        else self.chat_template
                    ),
                    "temperature": (
                        self.baseline_temperature
                        if self.baseline_temperature is not None
                        else self.temperature
                    ),
                    "top_p": (
                        self.baseline_top_p
                        if self.baseline_top_p is not None
                        else self.top_p
                    ),
                    "top_k": (
                        self.baseline_top_k
                        if self.baseline_top_k is not None
                        else self.top_k
                    ),
                    "seed": (
                        self.baseline_seed
                        if self.baseline_seed is not None
                        else self.seed
                    ),
                }
            )
        )
        return kwargs


class JudgeArgs(BaseModel):
    """The judge model and how it scores each battle."""

    model_config = ConfigDict(protected_namespaces=(), use_attribute_docstrings=True)

    model: str
    """LLM used as the judge, in ``{backend}/{model path}`` format (e.g.
    ``OpenRouter/deepseek/deepseek-chat-v3.1``)."""

    max_out_tokens: int = 32768
    """Generation token budget for the judge response (reasoning + scores)."""

    temperature: float | None = None
    """Sampling temperature for the judge model. Unset keeps the backend
    default, except MT-Bench FastChat-compatible judging which defaults to
    deterministic ``0.0``."""

    top_p: float | None = None
    """Nucleus-sampling probability for the judge model."""

    top_k: int | None = None
    """Top-k sampling cutoff for the judge model when supported."""

    seed: int | None = None
    """Backend sampling seed for the judge model when supported."""

    max_model_len: int | None = None
    """Optional total context window for the judge vLLM instance."""

    chat_template: str | None = None
    """Jinja2 chat template for the judge vLLM instance. Unset falls back to
    ``model.chat_template`` for backward compatibility."""

    engine_kwargs: dict = Field(default_factory=dict)
    """JSON dict of engine kwargs applied to the judge model only (overrides
    ``model.engine_kwargs`` for the judge)."""

    provide_explanation: bool = False
    """If set, the judge explains its reasoning before scoring. Aids
    interpretation; does not necessarily improve accuracy."""

    swap_mode: Literal["fixed", "both"] = "fixed"
    """Position-bias handling. ``fixed``: a single A-B judge pass. ``both``:
    judge each battle in both orders (A-B and B-A) and combine."""

    prompt_preset: str | None = None
    """Named judge prompt preset to use (see ``judgearena.prompts``). Defaults
    to the task's preset when unset."""

    system_prompt_file: str | None = None
    """Path to a custom judge system prompt, overriding the preset's system
    prompt."""

    user_prompt_file: str | None = None
    """Path to a custom judge user-prompt template, overriding the preset's."""

    battle_thinking_token_budget: int | None = None
    """Token budget allotted to a thinking judge's reasoning block. Unset
    leaves the model default."""

    strip_thinking_before_judging: bool = False
    """Strip ``<think>`` reasoning blocks from the battle completions before
    showing them to the judge."""

    def model_kwargs(
        self,
        *,
        base_engine_kwargs: dict | None = None,
        fallback_chat_template: str | None = None,
    ) -> dict[str, object]:
        """Kwargs for constructing the judge model."""
        kwargs: dict[str, object] = dict(base_engine_kwargs or {})
        kwargs.update(self.engine_kwargs)
        kwargs.update(
            _drop_none(
                {
                    "max_tokens": self.max_out_tokens,
                    "max_model_len": self.max_model_len,
                    "chat_template": (
                        self.chat_template
                        if self.chat_template is not None
                        else fallback_chat_template
                    ),
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "seed": self.seed,
                }
            )
        )
        return kwargs


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

    elo_random_battles: int | None = None
    """Sample N arena rows uniformly at random (seeded by ``run.seed``) instead
    of taking the first N. Mutually exclusive with the ``n_instructions*``
    caps."""

    soft_elo: bool = True
    """Use soft (continuous-preference) Bradley-Terry. When False, fall back to
    hard win/loss/tie labels."""

    soft_elo_temperature: float = 0.3
    """Initial PairScore temperature for soft-ELO. Overridden by
    ``calibrate_temperature`` when calibration succeeds."""

    calibrate_temperature: bool = False
    """MLE-fit the PairScore temperature against human-labeled arena battles
    before the main run. Ignored when ``soft_elo`` is False."""

    calibration_size: int | None = None
    """Number of human arena battles to sample for temperature calibration.
    Defaults to all. Requires ``calibrate_temperature``."""


class MetaEvalArgs(BaseModel):
    """Settings for judge meta-evaluation against human arena labels."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    reference_arena: str = "LMArena-140k"
    """Human-labeled reference arena to sample battles from."""

    prompt_mode: Literal[
        "standard",
        "arena-hard",
        "alpaca-eval",
        "alpaca-eval-pair-score",
    ] = "standard"
    """Named judge prompt and parser mode."""

    top_models: int = 20
    """Number of top models by battle count to include."""

    battles_per_model: int = 50
    """Battles sampled per top model."""

    batch_size: int = Field(default=50, gt=0)
    """Annotation batch size."""

    languages: list[str] | None = None
    """Restrict reference battles to ISO 639-1 language codes."""

    n_bootstraps: int = 20
    """Bootstrap samples used for uncertainty estimates."""

    elo_gap_battles: list[int] = Field(default_factory=lambda: [10, 20, 30, 40, 50])
    """Battle counts included in the ELO-gap analysis."""

    elo_gap_seeds: int = 10
    """Random seeds used for ELO-gap subsampling."""

    include_human_ties: bool = False
    """Include human-labeled ties in the primary agreement view."""


class CacheArgs(BaseModel):
    """Unified inference cache settings (SQLite cells with optional HF sync)."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    store_root: str | None = None
    """Local root for inference cache cells. When unset, caching is disabled."""

    cache_mode: CacheMode = "use"
    """``use``: read and insert rows. ``off``: always infer. ``refresh``: replace rows."""

    cache_hf_repo: str = DEFAULT_CACHE_REPO
    """Hugging Face dataset repo used when ``cache_fetch`` or ``cache_push`` is set."""

    cache_fetch: CliImplicitFlag[bool] = False
    """Explicit opt-in to fetch remote cache cells before inference."""

    cache_push: CliImplicitFlag[bool] = False
    """Explicit opt-in to push locally produced cache rows after a successful run."""

    cache_create_pr: CliImplicitFlag[bool] = False
    """Push cache updates through a Hugging Face pull request (requires ``cache_push``)."""

    pushed_by: str = Field(default_factory=lambda: default_pushed_by())
    """Provenance label recorded on locally produced cache rows."""

    @model_validator(mode="after")
    def _validate_cache_options(self) -> CacheArgs:
        if self.store_root is not None and not self.store_root.strip():
            raise ValueError("cache.store_root must be non-empty when provided.")
        if self.cache_fetch or self.cache_push or self.cache_create_pr:
            if not self.store_root:
                raise ValueError(
                    "cache.store_root is required when cache_fetch, cache_push, "
                    "or cache_create_pr is enabled."
                )
        if self.cache_fetch or self.cache_push:
            if not self.cache_hf_repo.strip():
                raise ValueError(
                    "cache.cache_hf_repo must be non-empty when cache_fetch or "
                    "cache_push is enabled."
                )
        if self.cache_create_pr and not self.cache_push:
            raise ValueError(
                "cache.cache_push is required when cache_create_pr is enabled."
            )
        if self.cache_mode == "off" and (self.cache_fetch or self.cache_push):
            raise ValueError(
                "cache_fetch and cache_push cannot be enabled when cache_mode is off."
            )
        if self.cache_mode == "refresh" and not self.store_root:
            raise ValueError("cache.store_root is required when cache_mode is refresh.")
        return self


class RunArgs(BaseModel):
    """Run-level settings: seed, output location, and logging."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    seed: int = 0
    """Random seed for reproducibility."""

    result_folder: str = "results"
    """Directory where annotations, results, and the resolved ``config.yaml``
    are written (under a per-run subfolder)."""

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
        cli_shortcuts=_CACHE_CLI_SHORTCUTS,
        use_attribute_docstrings=True,
    )

    task: str
    """Benchmark to run. Generate+judge: ``alpaca-eval``, ``arena-hard-v2.0``,
    ``m-arena-hard-*``, ``mt-bench``, ``fluency-*``. Meta-evaluation:
    ``meta-eval``. ELO: ``elo-lmarena-100k``, ``elo-lmarena-140k``,
    ``elo-lmarena``, ``elo-comparia``."""

    model: ModelArgs = Field(default_factory=ModelArgs)
    """Model(s) under evaluation and their generation settings."""

    judge: JudgeArgs
    """The judge model and scoring behaviour."""

    generation: GenerationArgs = Field(default_factory=GenerationArgs)
    """Instruction count and input truncation."""

    elo: EloArgs | None = None
    """ELO-task settings (only for ``elo-*`` tasks)."""

    meta_eval: MetaEvalArgs | None = None
    """Judge meta-evaluation settings (only for ``meta-eval``)."""

    run: RunArgs = Field(default_factory=RunArgs)
    """Run-level settings (seed, output, logging)."""

    cache: CacheArgs = Field(default_factory=CacheArgs)
    """Unified inference cache settings."""

    @model_validator(mode="after")
    def _validate(self) -> RunConfig:
        if self.task == META_EVAL_TASK:
            if self.meta_eval is None:
                self.meta_eval = MetaEvalArgs()
            if self.elo is not None:
                raise ValueError("elo config is only valid for elo-* tasks.")
            if self.model.name is not None or self.model.baseline is not None:
                raise ValueError(
                    "model config is not used for meta-eval; "
                    "only judge.model is required."
                )
            if self.model.max_out_tokens != ModelArgs().max_out_tokens:
                raise ValueError(
                    "model.max_out_tokens is not used for meta-eval because "
                    "no model completions are generated."
                )
            if self.generation.n_instructions is not None:
                raise ValueError(
                    "generation.n_instructions is not used for meta-eval; use "
                    "meta_eval.top_models and meta_eval.battles_per_model."
                )
            return self

        if self.meta_eval is not None:
            raise ValueError("meta_eval config is only valid for the meta-eval task.")

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


def meta_eval_cache_task(reference_arena: str) -> str:
    """Return the single-segment cache namespace for one meta-eval reference arena."""
    sanitized_arena = reference_arena.replace("/", "_").replace("\\", "_")
    return f"{META_EVAL_TASK}-{sanitized_arena}"


def inference_cache_task(cfg: RunConfig) -> str:
    """Return the cache namespace for a run configuration."""
    if cfg.task == META_EVAL_TASK:
        if cfg.meta_eval is None:
            raise ValueError("meta_eval config is required for the meta-eval task.")
        return meta_eval_cache_task(cfg.meta_eval.reference_arena)
    return cfg.task


@contextmanager
def open_inference_cache(
    cache_args: CacheArgs,
    task: str,
) -> Iterator[InferenceCache | None]:
    """Open a run-scoped inference cache, or yield ``None`` when disabled."""
    if cache_args.store_root is None:
        yield None
        return

    with InferenceCache(
        store_root=cache_args.store_root,
        task=task,
        mode=cache_args.cache_mode,
        fetch=cache_args.cache_fetch,
        push=cache_args.cache_push,
        create_pr=cache_args.cache_create_pr,
        cache_hf_repo=cache_args.cache_hf_repo,
        pushed_by=cache_args.pushed_by,
    ) as cache:
        yield cache


@contextmanager
def inference_cache_session(cfg: RunConfig) -> Iterator[InferenceCache | None]:
    """Open a run-scoped inference cache, or yield ``None`` when disabled."""
    with open_inference_cache(cfg.cache, inference_cache_task(cfg)) as cache:
        yield cache
