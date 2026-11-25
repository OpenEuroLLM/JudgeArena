"""Hierarchical run configuration (pydantic-settings) with YAML loading."""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    CliSuppress,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from judgearena.deprecations import parse_cli_overrides, translate_deprecations
from judgearena.log import get_logger
from judgearena.models import build_default_judge_model_kwargs
from judgearena.prompts.registry import prompt_option_files
from judgearena.prompts.spec import (
    JudgePromptSpec,
    config_prompts,
    resolve_prompt_paths,
)
from judgearena.tasks.registry import _canonical_sha256, get_packaged_task
from judgearena.tasks.schema import (
    AllReferencesBaseline,
    CategoryDefaultsBaseline,
    DatasetSpec,
    GraphArgs,
    MetaEvalProtocol,
    NoBaseline,
    ProtocolSpec,
    ResolvedTaskSpec,
    SingleAnswerProtocol,
    TaskDefaultBaseline,
    TaskProvenance,
    TaskSpec,
)
from judgearena.utils.extensions import resolve_component_paths

logger = get_logger(__name__)

# Set by build_run_config() for the duration of RunConfig() construction.
_ACTIVE_CONFIG_PATH: str | None = None
_ACTIVE_CLI_ARGS: list[str] | None = None


def _default_store_root() -> str:
    cache_home = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
    return str(cache_home.expanduser() / "judgearena")


def _drop_none(kwargs: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _merge_overrides(defaults: dict, overrides: dict, path: tuple = ()) -> dict:
    """Merge fields, replacing collections and sections whose implementation changes."""
    is_prompt = path[-1:] == ("prompt",) or path[-2:-1] == ("category_prompts",)
    if is_prompt and (
        any(key in overrides for key in ("file", "preset"))
        or (
            defaults.get("preset") is not None
            and any(key in overrides for key in ("user", "user_file"))
        )
    ):
        return overrides.copy()
    identity = {
        ("protocol",): "runner",
        ("dataset",): "adapter",
        ("protocol", "generation"): "mode",
        ("protocol", "baseline"): "strategy",
    }.get(path)
    if len(path) == 3 and path[:2] == ("dataset", "sources"):
        identity = "type"
    if identity in overrides and overrides[identity] != defaults.get(identity):
        return overrides.copy()
    merged = defaults.copy()
    if is_prompt:
        for field, opposite in (
            ("system", "system_file"),
            ("system_file", "system"),
            ("user", "user_file"),
            ("user_file", "user"),
        ):
            if overrides.get(field) is not None and opposite not in overrides:
                merged.pop(opposite, None)
    if path == ("protocol", "judge"):
        if (
            overrides.get("default_prompt_preset") is not None
            and "prompt" not in overrides
        ):
            merged.pop("prompt", None)
        elif overrides.get("prompt") is not None:
            merged.pop("default_prompt_preset", None)
    for key, value in overrides.items():
        if isinstance(value, dict) and value and isinstance(merged.get(key), dict):
            merged[key] = _merge_overrides(merged[key], value, (*path, key))
        else:
            merged[key] = value
    return merged


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

    sampling_kwargs: dict = Field(default_factory=dict)
    """Additional backend sampling options for each inference request."""

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

    baseline_sampling_kwargs: dict | None = None
    """Baseline sampling options. Unset inherits model.sampling_kwargs."""

    baseline_engine_kwargs: dict | None = None
    """JSON dict of engine-specific kwargs for the baseline/opponent model.
    Unset inherits ``model.engine_kwargs``."""

    def evaluated_generation_kwargs(self) -> dict[str, object]:
        """Kwargs for model A / the evaluated model."""
        kwargs: dict[str, object] = dict(self.engine_kwargs)
        if self.sampling_kwargs:
            kwargs["sampling_kwargs"] = self.sampling_kwargs
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
        sampling = (
            self.baseline_sampling_kwargs
            if self.baseline_sampling_kwargs is not None
            else self.sampling_kwargs
        )
        if sampling:
            kwargs["sampling_kwargs"] = sampling
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
    """The judge model, prompt, and inference settings."""

    model_config = ConfigDict(
        protected_namespaces=(), use_attribute_docstrings=True, extra="forbid"
    )

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

    sampling_kwargs: dict = Field(default_factory=dict)
    """Additional backend sampling options for each inference request."""

    engine_kwargs: dict = Field(default_factory=dict)
    """JSON dict of engine kwargs applied to the judge model only (overrides
    ``model.engine_kwargs`` for the judge)."""

    swap_mode: Literal["fixed", "both", "random"] = "fixed"
    """Position-bias handling. ``fixed``: a single A-B judge pass. ``both``:
    judge each battle in both orders (A-B and B-A) and combine. ``random``:
    a single pass with the pair order flipped deterministically per
    instruction (AlpacaEval's scheme)."""

    top_logprobs: int | None = None
    """Request the judge's top-N token logprobs (used by logprob-weighted
    annotators like AlpacaEval's). Unset requests none."""

    prompt: JudgePromptSpec | None = None
    """Prompt selection. When omitted, inherit the task's default preset."""

    battle_thinking_token_budget: int | None = None
    """Token budget allotted to a thinking judge's reasoning block. Unset
    leaves the model default."""

    strip_thinking_before_judging: bool = False
    """Strip ``<think>`` reasoning blocks from the battle completions before
    showing them to the judge."""

    @model_validator(mode="before")
    @classmethod
    def ignore_unused_legacy_fields(cls, values):
        if not isinstance(values, dict):
            return values
        values = values.copy()
        if values.get("provide_explanation") is False:
            values.pop("provide_explanation")
        for key in ("system_prompt_file", "user_prompt_file"):
            if values.get(key) is None:
                values.pop(key, None)
        return values

    def model_kwargs(
        self,
        *,
        base_engine_kwargs: dict | None = None,
        fallback_chat_template: str | None = None,
    ) -> dict[str, object]:
        """Kwargs for constructing the judge model."""
        kwargs: dict[str, object] = dict(base_engine_kwargs or {})
        kwargs.update(self.engine_kwargs)
        if self.sampling_kwargs:
            kwargs["sampling_kwargs"] = self.sampling_kwargs
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
                    "top_logprobs": self.top_logprobs,
                }
            )
        )
        return kwargs


class GenerationArgs(BaseModel):
    """How many instructions to use and input-length truncation."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    n_instructions: int | None = None
    """Number of instructions/battles to evaluate. Defaults to the full task."""

    truncate_all_input_chars: int | None = 8192
    """Character cap before model generation. None preserves the full instruction."""

    truncate_judge_input_chars: int | None = None
    """Character cap applied to judge-side inputs before evaluation. Unset
    means no judge-side character truncation."""


class MetaEvalArgs(BaseModel):
    """Sampling settings for judge meta-evaluation."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    top_models: int = Field(default=20, ge=2)
    """Number of the arena's most-battled models to include."""

    battles_per_model: int = Field(default=50, gt=0)
    """Minimum number of sampled incident battles for each selected model."""

    languages: list[str] | None = None
    """Restrict arena battles to these language codes. Defaults to all languages."""


class RunArgs(BaseModel):
    """Run-level settings: seed, output location, caching, and logging."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    seed: int = 0
    """Random seed for reproducibility."""

    filters: dict[str, list[str | int | float | bool]] = Field(default_factory=dict)
    """Select dataset rows by metadata values, e.g. {\"domain\": [\"math\"]}."""

    result_folder: str = "results"
    """Directory where annotations, results, and the resolved ``config.yaml``
    are written (under a per-run subfolder)."""

    store_root: str | None = Field(default_factory=_default_store_root)
    """Root directory for content-addressed caches. Set to null to disable."""

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

    task: str | None = None
    """Packaged benchmark task ID. Omit when supplying a protocol.
    Use ``judgearena tasks list`` to inspect available definitions."""

    evaluation_name: str = Field(
        default="custom", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
    )
    """Stable cache and report name for evaluations without a packaged task."""

    protocol: ProtocolSpec | None = None
    """Override task protocol fields; changing runner requires a complete protocol."""

    dataset: DatasetSpec | None = None
    """Override dataset fields; changing adapter replaces the declaration."""

    resolved_task: CliSuppress[ResolvedTaskSpec | None] = Field(
        default=None, repr=False
    )
    """Effective task and source provenance, retained when saving a configuration."""

    model: ModelArgs = Field(default_factory=ModelArgs)
    """Model(s) under evaluation and their generation settings."""

    judge: JudgeArgs
    """The judge model and scoring behaviour."""

    generation: GenerationArgs = Field(default_factory=GenerationArgs)
    """Instruction count and input truncation."""

    graph: GraphArgs | None = None
    """Sample saved comparisons instead of constructing baseline pairs."""

    meta_eval: MetaEvalArgs | None = None
    """Runtime settings used only by tasks with a meta-evaluation protocol."""

    run: RunArgs = Field(default_factory=RunArgs)
    """Run-level settings (seed, output, caching, logging)."""

    @model_validator(mode="before")
    @classmethod
    def _merge_task_overrides(cls, values):
        if not isinstance(values, dict):
            return values
        values = values.copy()
        task = values.get("resolved_task")
        if task is None and isinstance(values.get("task"), str):
            task = get_packaged_task(values["task"])
        definition = (
            task.get("spec") if isinstance(task, dict) else getattr(task, "spec", {})
        )
        if isinstance(definition, BaseModel):
            definition = definition.model_dump()
        values = translate_deprecations(values, definition.get("protocol", {}))
        for section in ("protocol", "dataset"):
            overrides = values.get(section)
            if isinstance(overrides, dict) and isinstance(
                definition.get(section), dict
            ):
                values[section] = _merge_overrides(
                    definition[section], overrides, (section,)
                )
        return values

    @model_validator(mode="after")
    def _validate(self) -> RunConfig:
        if self.judge.prompt is not None:
            prompt = self.judge.prompt
            for path in (
                prompt.system_file,
                prompt.user_file,
                *(
                    prompt.options.get(name)
                    for name in prompt_option_files(prompt.preset)
                ),
            ):
                if path is not None and (
                    not isinstance(path, (str, Path)) or not Path(path).is_file()
                ):
                    raise ValueError(f"judge prompt file not found: {path}")

        resolved_task = self.resolved_task
        if resolved_task is not None and (
            resolved_task.provenance.resolved_sha256
            != _canonical_sha256(resolved_task.spec.model_dump(mode="json"))
        ):
            logger.warning(
                "Resolved task %r has a stale definition hash; recomputing it from "
                "the supplied definition. Source provenance is unchanged.",
                resolved_task.task,
            )
        if resolved_task is None and self.task is not None:
            resolved_task = get_packaged_task(self.task)
            if resolved_task is None:
                raise ValueError(
                    f"Unknown task {self.task!r}; use 'judgearena tasks list' to "
                    "inspect packaged tasks."
                )
        if resolved_task is None:
            if self.protocol is None:
                raise ValueError("Provide a task or an evaluation protocol.")
            spec = TaskSpec(
                schema_version=1,
                task=self.evaluation_name,
                task_version=1,
                description="Inline evaluation.",
                dataset=self.dataset or DatasetSpec(adapter="supplied"),
                protocol=self.protocol,
            )
            digest = _canonical_sha256(spec.model_dump(mode="json"))
            resolved_task = ResolvedTaskSpec(
                spec, TaskProvenance("<inline>", digest, digest, ())
            )
        elif self.protocol is not None or self.dataset is not None:
            definition = resolved_task.spec.model_dump()
            if self.protocol is not None:
                definition["protocol"] = self.protocol.model_dump()
            if self.dataset is not None:
                definition["dataset"] = self.dataset.model_dump()
            spec = TaskSpec.model_validate(definition)
            resolved_task = replace(resolved_task, spec=spec)
        task_graph = getattr(resolved_task.spec.protocol, "graph", None)
        if self.graph is None and task_graph is not None:
            self.graph = task_graph.model_copy()
        resolved_task = replace(
            resolved_task,
            provenance=replace(
                resolved_task.provenance,
                resolved_sha256=_canonical_sha256(
                    resolved_task.spec.model_dump(mode="json")
                ),
            ),
        )
        if self.task is not None and resolved_task.task != self.task:
            raise ValueError("Resolved task must match run.task.")
        self.task = resolved_task.task
        self.resolved_task = resolved_task
        protocol = resolved_task.spec.protocol
        task_generation = getattr(protocol, "generation", None)
        if (
            "truncate_all_input_chars" not in self.generation.model_fields_set
            and not getattr(task_generation, "default_truncate_input", True)
        ):
            self.generation.truncate_all_input_chars = None
            self.generation.model_fields_set.discard("truncate_all_input_chars")
        model_fields = self.model.model_fields_set.copy()
        model_values = self.model.model_dump(exclude_unset=True)
        for field, engine_key in (
            ("temperature", "temperature"),
            ("max_out_tokens", "max_tokens"),
            ("seed", "seed"),
        ):
            default = getattr(task_generation, f"default_{field}", None)
            if default is None:
                continue
            if field not in self.model.model_fields_set:
                model_values[field] = self.model.engine_kwargs.get(engine_key, default)

            baseline_field = f"baseline_{field}"
            if getattr(self.model, baseline_field) is not None:
                continue
            # Dedicated model settings still take precedence when inherited.
            if (
                field not in self.model.model_fields_set
                and self.model.baseline_engine_kwargs is not None
                and engine_key in self.model.baseline_engine_kwargs
            ):
                model_values[baseline_field] = self.model.baseline_engine_kwargs[
                    engine_key
                ]
        self.model = ModelArgs.model_validate(model_values)
        self.model.model_fields_set.intersection_update(model_fields)

        task_judge = protocol.judge
        judge_values = self.judge.model_dump(exclude_unset=True)
        judge_fields = self.judge.model_fields_set.copy()
        if "swap_mode" not in judge_values:
            judge_values["swap_mode"] = getattr(
                task_judge, "default_swap_mode", "fixed"
            )
        judge_engine_kwargs = self.judge_engine_kwargs()
        for field, engine_key in (
            ("temperature", "temperature"),
            ("max_out_tokens", "max_tokens"),
            ("top_logprobs", "top_logprobs"),
        ):
            default = getattr(task_judge, f"default_{field}")
            if default is None or field in judge_values:
                continue
            judge_values[field] = judge_engine_kwargs.get(engine_key, default)
        self.judge = JudgeArgs.model_validate(judge_values)
        self.judge.model_fields_set.intersection_update(judge_fields)

        is_meta_eval = isinstance(protocol, MetaEvalProtocol)
        if self.meta_eval is not None and not is_meta_eval:
            raise ValueError(
                "meta_eval config is only valid for meta-evaluation tasks."
            )
        if is_meta_eval:
            if self.meta_eval is None:
                self.meta_eval = MetaEvalArgs()
            if self.model.name is not None or self.model.baseline is not None:
                raise ValueError(
                    "model.name and model.baseline are not supported for "
                    "meta-evaluation tasks."
                )
            if self.judge.swap_mode == "random":
                raise ValueError(
                    "judge.swap_mode='random' is not supported for "
                    "meta-evaluation tasks."
                )
        else:
            if self.model.name is None:
                raise ValueError("model.name is required.")
            if self.graph is not None and self.model.baseline is not None:
                raise ValueError("model.baseline is not supported with graph sampling.")
            if isinstance(protocol, SingleAnswerProtocol):
                if self.model.baseline is not None or self.judge.swap_mode != "fixed":
                    raise ValueError(
                        "Independent single-answer evaluation does not use a baseline or swapped judging."
                    )
            elif isinstance(protocol.baseline, AllReferencesBaseline):
                if self.model.name in protocol.baseline.references:
                    raise ValueError(
                        "model.name must differ from the all_references models."
                    )
                if self.model.baseline is not None:
                    raise ValueError(
                        "model.baseline cannot override an all_references policy."
                    )
            elif self.model.baseline is None and not isinstance(
                protocol.baseline,
                (TaskDefaultBaseline, CategoryDefaultsBaseline, NoBaseline),
            ):
                raise ValueError(f"model.baseline is required for task {self.task!r}.")
        return self

    def judge_engine_kwargs(self) -> dict[str, object]:
        """Resolve engine inheritance and defaults for this task's judge."""
        assert self.resolved_task is not None
        if not getattr(
            self.resolved_task.spec.protocol.judge, "use_default_engine_kwargs", True
        ):
            return {**self.model.engine_kwargs, **self.judge.engine_kwargs}
        return build_default_judge_model_kwargs(
            self.judge.model,
            self.model.engine_kwargs,
            judge_engine_kwargs_override=self.judge.engine_kwargs,
        )

    def judge_model_kwargs(self) -> dict[str, object]:
        """Effective judge settings for prompt validation and model construction."""
        return self.judge.model_kwargs(
            base_engine_kwargs=self.judge_engine_kwargs(),
            fallback_chat_template=self.model.chat_template,
        )

    @property
    def preference_settings(self):
        """Judge-owned preference conversion settings."""
        return self.resolved_task.spec.protocol.judge

    def for_task(self, task: ResolvedTaskSpec | None) -> RunConfig:
        """Apply task defaults, retaining explicit settings (all saved settings on reload)."""
        if task is None or task == self.resolved_task:
            return self
        values = self.model_dump(exclude={"resolved_task"})
        for name in values:
            value = getattr(self, name)
            if isinstance(value, BaseModel):
                values[name] = value.model_dump(exclude_unset=True)
        values["model"]["engine_kwargs"] = self.model.engine_kwargs.copy()
        values["judge"]["engine_kwargs"] = self.judge.engine_kwargs.copy()
        values["task"] = task.task
        return type(self)(**values, resolved_task=task)

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
        if _ACTIVE_CLI_ARGS is None and _ACTIVE_CONFIG_PATH is None:
            return (init_settings,)
        values = resolve_cli_config(
            _ACTIVE_CLI_ARGS or [],
            config_path=_ACTIVE_CONFIG_PATH,
            settings_cls=settings_cls,
        )
        return (InitSettingsSource(settings_cls, values),)


def resolve_cli_config(
    argv, *, config_path=None, settings_cls=RunConfig, allowed_sections=None
):
    """Resolve task/config/CLI values before command-specific model validation."""
    overrides = parse_cli_overrides(settings_cls, argv)
    values = (
        _read_config(
            config_path,
            task_override=overrides.get("task"),
            cli_elo=overrides.get("elo"),
        )
        if config_path
        else {}
    )
    if "task" in overrides:
        values = {**values, "task": overrides["task"]}
    values = settings_cls._merge_task_overrides(values)
    if "elo" in overrides:
        if "languages" in (overrides["elo"] or {}) and values.get("run"):
            overrides["run"] = _merge_overrides(values["run"], overrides.get("run", {}))
        protocol = values.get("protocol")
        if protocol is None:
            protocol = settings_cls._merge_task_overrides({**values, "protocol": {}})[
                "protocol"
            ]
        overrides = translate_deprecations(overrides, protocol, cli=True)
    if allowed_sections is not None and (set(values) | set(overrides)) - set(
        allowed_sections
    ):
        raise ValueError(
            f"Command accepts {', '.join(sorted(allowed_sections))} settings only."
        )
    return _merge_overrides(values, overrides)


def build_run_config(argv: list[str] | None = None) -> RunConfig:
    """Build a RunConfig from CLI flags and an optional --config_path YAML.

    Precedence: CLI flags > --config_path YAML > model defaults.
    """
    global _ACTIVE_CONFIG_PATH, _ACTIVE_CLI_ARGS
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config_path", "--config", default=None)
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


def _migrate_saved_run(data: dict) -> dict:
    """Read old run snapshots without exposing their options to new experiments."""
    resolved = data.get("resolved_task")
    if not isinstance(resolved, dict):
        return data
    data = deepcopy(data)
    resolved = data["resolved_task"]
    spec = resolved["spec"]
    protocol = spec["protocol"]
    saved_format = spec["dataset"].get("format")
    for dataset in (spec["dataset"], data.get("dataset") or {}):
        if not ({"format", "battle_columns"} & dataset.keys()) and dataset.get(
            "adapter"
        ) not in {"arena_battles", "judgearena_tables"}:
            continue
        release = dataset.pop("format", None)
        if dataset.pop("battle_columns", {}) or release == "battles":
            raise ValueError(
                "Saved battle-table configs require a custom dataset adapter."
            )
        if dataset.get("adapter") == "judgearena_tables" and spec["task"] in {
            "alpaca-eval",
            "alpaca-eval-ja",
        }:
            dataset["adapter"] = "alpaca_eval"
        if dataset.get("adapter") == "arena_battles":
            release = release or saved_format or protocol.get("arena")
            if release == "ComparIA":
                dataset["adapter"] = "comparia"
            elif release in {"LMArena", "LMArena-55k", "LMArena-100k", "LMArena-140k"}:
                dataset["adapter"] = "lmarena"
            else:
                raise ValueError(f"Unknown saved arena format {release!r}")
        # A saved release selector could narrow a combined source list.
        if dataset.get("adapter") == "lmarena" and release and release != "LMArena":
            repo_id = (
                f"lmarena-ai/arena-human-preference-{release.removeprefix('LMArena-')}"
            )
            dataset["sources"] = {
                key: source
                for key, source in dataset["sources"].items()
                if source.get("repo_id") == repo_id
            }
            if not dataset["sources"]:
                raise ValueError(
                    f"No declared dataset source matches saved format {release!r}."
                )
        resolved["provenance"]["resolved_sha256"] = _canonical_sha256(spec)
    for graph in (
        data.get("graph"),
        (data.get("protocol") or {}).get("graph"),
        protocol.get("graph"),
    ):
        if isinstance(graph, dict) and "per_language" in graph:
            quota = graph.pop("per_language")
            if quota is not None:
                graph.setdefault("group_by", "lang")
                graph.setdefault("per_group", quota)
    options = data.pop("elo", None) or {}
    old_runner = protocol["runner"] == "elo"
    if not old_runner and not options:
        return data
    if protocol["runner"] not in {"elo", "pairwise"}:
        raise ValueError("Saved Elo settings require a pairwise task")

    scoring = protocol["scoring"]
    metrics = scoring["metrics"]
    defaults = next(
        (
            m.get("parameters", {})
            for m in metrics
            if m["metric"] == "bradley_terry"
            and m.get("name") in (None, "bradley_terry")
        ),
        {},
    )
    soft = options.get(
        "soft_elo", defaults.get("soft", scoring.get("default_soft", True))
    )
    temperature = options.get(
        "soft_elo_temperature", scoring.get("default_temperature", 0.3)
    )
    # Before frozen fitting existed, omitted fit_mode meant a joint fit.
    fit_mode = options.get("fit_mode", defaults.get("fit_mode", "joint"))
    graph = data.get("graph") or protocol.get("graph") or {}
    for old, new in (
        ("elo_random_battles", "random_battles"),
        ("n_instructions_per_language", "per_group"),
    ):
        if options.get(old) is not None:
            graph.setdefault(new, options[old])
    if graph.get("per_group") is not None:
        graph.setdefault("group_by", "lang")
    protocol.update(runner="pairwise", graph=graph)
    data["graph"] = graph
    protocol.pop("arena", None)
    judge = protocol["judge"]
    for key, value in {
        "preference_temperature": temperature if soft else None,
        "calibrate_temperature": options.get("calibrate_temperature", False) and soft,
        "calibration_size": options.get("calibration_size"),
    }.items():
        if judge.get(key) is None:
            judge[key] = value
    for metric in metrics:
        if metric["metric"] != "bradley_terry":
            continue
        parameters = metric.setdefault("parameters", {})
        migrated = {
            "fit_mode": fit_mode,
            "soft": soft,
            "baseline_model": options.get(
                "baseline_model", defaults.get("baseline_model")
            ),
            "n_bootstraps": options.get(
                "n_bootstraps", defaults.get("n_bootstraps", 20)
            ),
        }
        # The old runner applied its runtime overrides to the default metric only.
        if metric.get("name") in (None, "bradley_terry"):
            parameters.update(migrated)
        else:
            for key, value in migrated.items():
                parameters.setdefault(key, value)
        parameters.setdefault(
            "reference_min_battles", 500 if parameters["fit_mode"] == "joint" else 0
        )
    scoring.pop("default_soft", None)
    scoring.pop("default_temperature", None)
    if options.get("languages"):
        from judgearena.datasets import resolve_dataset_filters

        run = data.setdefault("run", {})
        run["filters"] = resolve_dataset_filters(
            None, run.get("filters", {}), {"lang": options["languages"]}
        )
    data.update(protocol=protocol, dataset=spec["dataset"])
    data["resolved_task"]["provenance"]["resolved_sha256"] = _canonical_sha256(spec)
    return data


def _read_config(
    path: str | Path, *, task_override: str | None = None, cli_elo: dict | None = None
) -> dict:
    """Read YAML, resolving component and prompt paths beside the configuration file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping.")
    data = _migrate_saved_run(data)
    # Apply hard mode after both sources have supplied their legacy judge settings.
    file_elo = data.get("elo")
    if isinstance(file_elo, dict) and cli_elo and "languages" in cli_elo:
        file_elo.pop("languages", None)
    if (
        isinstance(file_elo, dict)
        and "soft_elo" in file_elo
        and cli_elo
        and (
            cli_elo.keys()
            & {"soft_elo", "soft_elo_temperature", "calibrate_temperature"}
        )
    ):
        cli_elo.setdefault("soft_elo", file_elo.pop("soft_elo"))
    if task_override is not None and task_override != data.get("task"):
        data["task"] = task_override
        data.pop("resolved_task", None)
    resolved = data.get("resolved_task")
    hash_matches = isinstance(resolved, dict) and resolved.get("provenance", {}).get(
        "resolved_sha256"
    ) == _canonical_sha256(resolved.get("spec", {}))
    resolve_component_paths(data, Path(path).resolve().parent)
    data = RunConfig._merge_task_overrides(data)
    for prompt in config_prompts(data):
        resolve_prompt_paths(prompt, Path(path).resolve().parent)
    if hash_matches:
        # Path resolution changes the definition, not its source provenance.
        resolved["provenance"]["resolved_sha256"] = _canonical_sha256(resolved["spec"])
    return data


def load_config(path: str | Path) -> RunConfig:
    """Load YAML; relative component and prompt paths start beside the YAML file."""
    return RunConfig(**_read_config(path))


def dump_config(cfg: RunConfig, path: str | Path) -> None:
    """Write the resolved config as YAML (round-trippable via ``--config_path``)."""
    Path(path).write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
