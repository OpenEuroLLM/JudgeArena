from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import partial
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from judgearena.prompts.criteria import configure_criteria_prompt
from judgearena.utils.extensions import load_task_components, loading_component

if TYPE_CHECKING:
    from judgearena.config import RunConfig
    from judgearena.tasks.schema import ResolvedTaskSpec

from judgearena.prompts.parsing import (
    JUDGE_PARSERS,
    OutputParser,
    ParsedPreference,
    parser_name,
    resolve_judge_parser,
    validate_parser,
)

PromptSource = Literal["preset", "file", "override", "delegated"]
JudgingMode = Literal["pairwise", "single_answer"]

DEFAULT_JUDGE_PROMPT_PRESET = "default"
DEFAULT_WITH_EXPLANATION_PRESET = "default_with_explanation"

# Legacy Python imports; task registration uses its own preset names.
FLUENCY_JUDGE_PROMPT_PRESET = "fluency"
FASTCHAT_PAIRWISE_PROMPT_PRESET = "fastchat-pairwise"
MT_BENCH_101_PROMPT_PRESET = "mt-bench-101"
MT_BENCH_101_CLEAN_PROMPT_PRESET = "mt-bench-101-clean"
ARENA_HARD_JUDGE_PROMPT_PRESET = "arena-hard"
ARENA_HARD_CREATIVE_JUDGE_PROMPT_PRESET = "arena-hard-creative"
ALPACA_EVAL_JUDGE_PROMPT_PRESET = "alpaca-eval"
META_EVAL_PAIR_SCORE_PROMPT_PRESET = "meta-eval-pair-score"

PROMPTS_PACKAGE = "judgearena.prompts"
_COMPLETION_LABEL_SINGLE = "Answer"
_COMPLETION_LABEL_MULTI_TURN = "Conversation with User"
_EXPLANATION_SUFFIX = ", first starts with an explanation of your judgement"
_SCORE_FENCE = "\n```"


@dataclass(frozen=True)
class JudgePromptPreset:
    name: str
    parser: OutputParser | None = None
    """Parser for this preset; None when delegated or configured during resolution."""
    system_file: str | None = None
    user_file: str | None = None
    inline_system: str | None = None
    inline_user: str | None = None
    delegated: bool = False
    with_explanation: bool = False
    output_format_file: str | None = None
    aggregation: Literal["individual", "agreement"] = "individual"
    prepare_prompts: Callable | None = None
    """Optional (cfg, prompt, instructions) -> (batch preparation, temperature)."""
    score_renderer: Callable | None = None
    """Optional single-answer renderer: (samples, prompt_preset, truncate_input_chars,
    strip_thinking_before_judging) -> (rendering fields, judge requests, prompts).
    Rows and requests follow the sample index; see judge_single_answers.
    """
    judging_mode: JudgingMode = "pairwise"
    """Number of candidates the prompt judges, independent of the task's runner."""

    configure: Callable[..., ResolvedJudgePrompt] | None = None
    """Optional (resolved_prompt, **options) -> resolved_prompt; owns option validation."""
    option_files: tuple[str, ...] = ()
    """File-valued option names to resolve beside config YAML and snapshot for replay."""

    bundle_spec: dict | None = field(default=None, repr=False, compare=False)
    """Expanded YAML settings retained for portable replay of registered bundles."""

    @classmethod
    def from_file(cls, name: str, path: str | Path) -> JudgePromptPreset:
        """Build a named preset using the same YAML schema as tasks and configs."""
        from judgearena.prompts.spec import JudgePromptSpec

        bundle = JudgePromptSpec(file=path)
        if bundle.preset is not None:
            base = PRESETS[bundle.preset]
            if bundle.options and base.configure is None:
                raise ValueError(
                    f"Prompt preset {bundle.preset!r} does not accept options."
                )
            return replace(
                base,
                name=name,
                bundle_spec=bundle.model_dump(mode="json"),
                configure=partial(base.configure, **bundle.options)
                if bundle.options
                else base.configure,
            )
        parser = resolve_judge_parser(bundle.parser or "score")
        return cls(
            name=name,
            bundle_spec=bundle.model_dump(mode="json"),
            parser=parser,
            system_file=str(bundle.system_file) if bundle.system_file else None,
            user_file=str(bundle.user_file) if bundle.user_file else None,
            inline_system=bundle.system,
            inline_user=bundle.user,
            judging_mode=bundle.judging_mode,
        )


@dataclass(frozen=True)
class ResolvedJudgePrompt:
    preset_name: str
    parser: OutputParser | None
    system_prompt: str | None
    user_prompt_template: str
    source: PromptSource
    system_path: str | None = None
    user_path: str | None = None
    system_sha256: str | None = None
    user_sha256: str | None = None
    delegated: bool = False

    variant_name: str | None = None
    output_format: str | None = None
    aggregation: Literal["individual", "agreement"] = "individual"
    prepare_prompts: Callable | None = None
    score_renderer: Callable | None = None
    judging_mode: JudgingMode = "pairwise"

    def validate_for(
        self,
        mode: JudgingMode,
        result_type: type | None = None,
        *,
        inference_kwargs: dict[str, object] | None = None,
    ) -> None:
        """Check the operation and supplied model settings; defer unresolved parsers."""
        if self.judging_mode != mode:
            raise ValueError(
                f"Prompt {self.preset_name!r} has judging mode {self.judging_mode!r}; "
                f"requires {mode!r}."
            )
        if self.parser is None:
            if self.delegated:
                return
            raise ValueError(f"Prompt {self.preset_name!r} requires a parser.")
        validate_parser(
            self.parser, ParsedPreference if mode == "pairwise" else result_type
        )
        if inference_kwargs is not None and self.parser.requires_top_logprobs:
            count = inference_kwargs.get("top_logprobs")
            if type(count) is not int or count <= 0:
                raise ValueError(
                    f"Prompt {self.preset_name!r} requires a positive integer "
                    "top_logprobs setting."
                )

    def metadata(self) -> dict[str, str | bool | None]:
        return {
            "judge_prompt_preset": self.preset_name,
            "judge_prompt_mode": self.judging_mode,
            **(
                {"judge_prompt_variant": self.variant_name} if self.variant_name else {}
            ),
            "judge_parser": (
                parser_name(self.parser) if self.parser is not None else None
            ),
            "judge_prompt_source": self.source,
            "judge_prompt_delegated": self.delegated,
            "judge_aggregation": self.aggregation,
            "judge_prompt_system_path": self.system_path,
            "judge_prompt_user_path": self.user_path,
            "judge_prompt_system_sha256": self.system_sha256,
            "judge_prompt_user_sha256": self.user_sha256,
        }


def build_chat_prompt_template(
    prompt: ResolvedJudgePrompt,
) -> ChatPromptTemplate:
    message_templates: list[tuple[str, str]] = []
    if prompt.system_prompt is not None:
        message_templates.append(("system", prompt.system_prompt))
    message_templates.append(("user", prompt.user_prompt_template))
    return ChatPromptTemplate.from_messages(message_templates)


def prompt_metadata(
    prompts: Iterable[ResolvedJudgePrompt],
) -> list[dict[str, str | bool | None]]:
    """Describe distinct selected prompts in first-use order."""
    variants = []
    for prompt in prompts:
        metadata = {
            **prompt.metadata(),
            "system_prompt": prompt.system_prompt,
            "user_prompt_template": prompt.user_prompt_template,
        }
        if metadata not in variants:
            variants.append(metadata)
    return variants


SCORE_PARSER = JUDGE_PARSERS["score"]

PRESETS: dict[str, JudgePromptPreset] = {
    "single-score": JudgePromptPreset(
        name="single-score",
        judging_mode="single_answer",
        parser=JUDGE_PARSERS["single-score"],
        user_file="single-answer.txt",
        output_format_file="single-score.txt",
    ),
    "criteria-score": JudgePromptPreset(
        name="criteria-score",
        judging_mode="single_answer",
        user_file="single-answer.txt",
        output_format_file="single-criteria.txt",
        configure=configure_criteria_prompt,
        option_files=("criteria_file",),
    ),
    "criteria": JudgePromptPreset(
        name="criteria",
        system_file="system-prompt.txt",
        user_file="prompt.txt",
        output_format_file="prompt-criteria.txt",
        configure=configure_criteria_prompt,
        option_files=("criteria_file",),
    ),
    DEFAULT_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=DEFAULT_JUDGE_PROMPT_PRESET,
        parser=SCORE_PARSER,
        system_file="system-prompt.txt",
        user_file="prompt.txt",
        output_format_file="score-output.txt",
    ),
    DEFAULT_WITH_EXPLANATION_PRESET: JudgePromptPreset(
        name=DEFAULT_WITH_EXPLANATION_PRESET,
        parser=SCORE_PARSER,
        system_file="system-prompt.txt",
        user_file="prompt.txt",
        output_format_file="score-output.txt",
        with_explanation=True,
    ),
}

JUDGE_PROMPT_PRESETS = PRESETS.keys()


def available_prompt_presets() -> tuple[str, ...]:
    """Return prompt IDs registered in this process, including user bundles."""
    load_task_components()
    return tuple(PRESETS)


def register_prompt(preset: JudgePromptPreset) -> JudgePromptPreset:
    """Register a prompt/parser bundle before constructing or loading a task."""
    if not preset.name.strip():
        raise ValueError("Prompt preset names must not be empty.")
    if preset.inline_user is not None and preset.user_file is not None:
        raise ValueError("Choose inline_user or user_file for a prompt preset.")
    if loading_component():
        return preset
    if preset.name in PRESETS:
        raise ValueError(f"Prompt preset {preset.name!r} is already registered.")
    PRESETS[preset.name] = preset
    return preset


def prompt_option_files(preset: str | None) -> tuple[str, ...]:
    """Return file options declared by a registered preset."""
    load_task_components()
    spec = PRESETS.get(preset) if isinstance(preset, str) else None
    return spec.option_files if spec is not None else ()


def default_preset_for_task(task: str | ResolvedTaskSpec | None) -> str:
    if task is None:
        return DEFAULT_JUDGE_PROMPT_PRESET
    if not isinstance(task, str):
        return (
            task.spec.protocol.judge.prompt.preset
            if task.spec.protocol.judge.prompt is not None
            else task.spec.protocol.judge.default_prompt_preset
        ) or DEFAULT_JUDGE_PROMPT_PRESET
    # Import lazily: task validation uses the prompt catalog in this module.
    from judgearena.tasks.registry import get_packaged_task

    resolved = get_packaged_task(task)
    if resolved is not None:
        return default_preset_for_task(resolved)
    return DEFAULT_JUDGE_PROMPT_PRESET


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_packaged_text(filename: str) -> str:
    return (
        files(PROMPTS_PACKAGE)
        .joinpath("templates", filename)
        .read_text(encoding="utf-8")
    )


def _materialize_user_template(
    text: str, *, multi_turn: bool, with_explanation: bool
) -> str:
    text = text.replace(
        "{completion_label}",
        _COMPLETION_LABEL_MULTI_TURN if multi_turn else _COMPLETION_LABEL_SINGLE,
    )
    text = text.replace(
        "{explanation_suffix}",
        _EXPLANATION_SUFFIX if with_explanation else _SCORE_FENCE,
    )
    return text


def _resolve_custom_prompt(
    *,
    system_file: str | Path | None,
    user_file: str | Path | None,
    user: str | None,
    system: str | None,
    multi_turn: bool,
    parser: OutputParser,
    judging_mode: JudgingMode,
) -> ResolvedJudgePrompt:
    system_path = Path(system_file) if system_file is not None else None
    user_path = Path(user_file) if user_file is not None else None
    system_prompt = (
        system_path.read_text(encoding="utf-8") if system_path is not None else system
    )
    user_prompt_template = _materialize_user_template(
        user_path.read_text(encoding="utf-8") if user_path is not None else user,
        multi_turn=multi_turn,
        with_explanation=False,
    )
    return ResolvedJudgePrompt(
        preset_name=(
            f"file:{system_path.name}+{user_path.name}"
            if system_path
            else f"file:{user_path.name}"
        )
        if user_path
        else "inline",
        parser=parser,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        source="file" if user_path else "override",
        judging_mode=judging_mode,
        system_path=str(system_path) if system_path is not None else None,
        user_path=str(user_path) if user_path else None,
        system_sha256=_sha256(system_prompt) if system_prompt is not None else None,
        user_sha256=_sha256(user_prompt_template),
    )


def resolve_judge_prompt(
    *,
    task: str | ResolvedTaskSpec | None = None,
    preset: str | None = None,
    system_file: str | Path | None = None,
    user_file: str | Path | None = None,
    user: str | None = None,
    system: str | None = None,
    multi_turn: bool = False,
    parser: str | None = None,
    options: dict[str, object] | None = None,
    judging_mode: JudgingMode = "pairwise",
) -> ResolvedJudgePrompt:
    """Load a preset or custom templates with a caller-supplied judging mode.

    Presets own their mode and parser; custom templates use the requested mode and
    named parser. Consumers validate compatibility before generating answers.
    """
    load_task_components()
    if (
        preset is None
        and task is not None
        and all(
            value is None for value in (user_file, user, system_file, system, parser)
        )
    ):
        if isinstance(task, str):
            from judgearena.tasks.registry import get_packaged_task

            task = get_packaged_task(task)
        if task is not None and task.spec.protocol.judge.prompt is not None:
            selection = task.spec.protocol.judge.prompt
            if options is not None:
                selection = selection.model_copy(update={"options": options})
            return resolve_prompt_spec(selection, multi_turn=multi_turn)
    if preset is None:
        preset = default_preset_for_task(task)
    if user_file is not None and user is not None:
        raise ValueError("Select either a user template or a user file.")
    if system_file is not None and system is not None:
        raise ValueError("Select either a system template or a system file.")
    if (
        (system_file is not None or system is not None)
        and user_file is None
        and user is None
    ):
        raise ValueError("A judge user template is required with a system template.")
    if user_file is not None or user is not None:
        if options:
            raise ValueError("Prompt options require a preset.")
        return _resolve_custom_prompt(
            system_file=system_file,
            user_file=user_file,
            user=user,
            system=system,
            multi_turn=multi_turn,
            parser=resolve_judge_parser(parser or "score"),
            judging_mode=judging_mode,
        )
    if parser is not None:
        raise ValueError(
            "judge.parser requires judge prompt files; a preset already "
            "defines its parser."
        )

    spec = PRESETS.get(preset)
    if spec is None:
        raise KeyError(
            f"Unknown judge prompt preset {preset!r}. Available: {sorted(PRESETS)}"
        )

    system_prompt = None
    user_prompt_template = ""
    output_format = None
    if not spec.delegated:
        if spec.user_file is None and spec.inline_user is None:
            raise ValueError(
                f"Judge prompt preset {spec.name!r} requires a user template."
            )
        system_prompt = (
            spec.inline_system
            if spec.inline_system is not None
            else _load_packaged_text(spec.system_file)
            if spec.system_file is not None
            else None
        )
        output_format = (
            _materialize_user_template(
                _load_packaged_text(spec.output_format_file),
                multi_turn=multi_turn,
                with_explanation=spec.with_explanation,
            )
            if spec.output_format_file is not None
            else None
        )
        user_prompt_template = _materialize_user_template(
            spec.inline_user
            if spec.inline_user is not None
            else _load_packaged_text(spec.user_file),
            multi_turn=multi_turn,
            with_explanation=spec.with_explanation,
        )
        if output_format is not None:
            user_prompt_template = (
                user_prompt_template.rstrip("\n") + "\n\n" + output_format
            )
    prompt = ResolvedJudgePrompt(
        preset_name=spec.name,
        parser=spec.parser,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        source="delegated" if spec.delegated else "preset",
        delegated=spec.delegated,
        system_path=spec.system_file,
        user_path=spec.user_file,
        output_format=output_format,
        aggregation=spec.aggregation,
        prepare_prompts=spec.prepare_prompts,
        score_renderer=spec.score_renderer,
        judging_mode=spec.judging_mode,
    )
    if spec.configure is not None:
        prompt = spec.configure(prompt, **(options or {}))
    elif options:
        raise ValueError(f"Prompt preset {preset!r} does not accept options.")
    return replace(
        prompt,
        system_sha256=_sha256(prompt.system_prompt)
        if prompt.system_prompt is not None
        else None,
        user_sha256=_sha256(prompt.user_prompt_template)
        if not prompt.delegated
        else None,
    )


def resolve_run_judge_prompt(
    task: str | ResolvedTaskSpec | None,
    judge_cfg,
    *,
    multi_turn: bool = False,
) -> ResolvedJudgePrompt:
    if isinstance(task, str):
        from judgearena.tasks.registry import get_packaged_task

        task = get_packaged_task(task)
    prompt = getattr(judge_cfg, "prompt", None)
    if prompt is None and task is not None:
        prompt = task.spec.protocol.judge.prompt
    if prompt is not None:
        return resolve_prompt_spec(prompt, multi_turn=multi_turn)
    return resolve_judge_prompt(task=task, multi_turn=multi_turn)


def resolve_prompt_spec(prompt, *, multi_turn: bool = False) -> ResolvedJudgePrompt:
    """Resolve the shared task/config selection through the existing registry."""
    resolved = resolve_judge_prompt(
        judging_mode=prompt.judging_mode or "pairwise",
        options=prompt.options,
        preset=prompt.preset,
        system_file=prompt.system_file,
        user_file=prompt.user_file,
        user=prompt.user,
        system=prompt.system,
        multi_turn=multi_turn,
        parser=prompt.parser,
    )
    return replace(resolved, preset_name=prompt.name) if prompt.name else resolved


def group_judge_prompts(
    cfg: RunConfig,
    instructions_df: pd.DataFrame,
    resolved_prompt: ResolvedJudgePrompt,
) -> list[tuple[ResolvedJudgePrompt, pd.Index]]:
    """Group evaluation rows that share the same resolved judge prompt.

    Tasks may map categories to prompt presets (e.g. Arena-Hard v2.0 judges
    creative writing with a different system prompt). Runtime prompt overrides
    (a different preset, preset options, or prompt files) disable the per-category mapping.
    """
    judge = cfg.resolved_task.spec.protocol.judge
    category_prompts = judge.category_prompts
    default_preset = (
        (judge.prompt.name or judge.prompt.preset)
        if judge.prompt is not None
        else judge.default_prompt_preset
    )
    override = cfg.judge.prompt
    if (
        not category_prompts
        or "category" not in instructions_df.columns
        or (
            override is not None
            and (
                override.options
                or (override.name or override.preset) != default_preset
                or (override.name or override.preset) is None
            )
        )
    ):
        return [(resolved_prompt, instructions_df.index)]

    categories = instructions_df["category"]
    groups: list[tuple[ResolvedJudgePrompt, pd.Index]] = []
    for category, rows in categories.groupby(
        categories, sort=False, dropna=False, observed=True
    ):
        preset = None if pd.isna(category) else category_prompts.get(category)
        prompt = (
            resolved_prompt
            if preset is None
            else resolve_judge_prompt(preset=preset)
            if isinstance(preset, str)
            else resolve_prompt_spec(preset)
        )
        groups.append((prompt, rows.index))
    return groups
