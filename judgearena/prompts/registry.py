from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

from judgearena.prompts.parsing import (
    JUDGE_PARSERS,
    JudgeParser,
    parser_name,
    resolve_judge_parser,
)

PromptSource = Literal["preset", "file", "task", "override", "delegated"]

DEFAULT_JUDGE_PROMPT_PRESET = "default"
DEFAULT_WITH_EXPLANATION_PRESET = "default_with_explanation"
FLUENCY_JUDGE_PROMPT_PRESET = "fluency"
FASTCHAT_PAIRWISE_PROMPT_PRESET = "fastchat-pairwise"
ARENA_HARD_JUDGE_PROMPT_PRESET = "arena-hard"
ARENA_HARD_CREATIVE_JUDGE_PROMPT_PRESET = "arena-hard-creative"

PROMPTS_PACKAGE = "judgearena.prompts"
_COMPLETION_LABEL_SINGLE = "Answer"
_COMPLETION_LABEL_MULTI_TURN = "Conversation with User"
_EXPLANATION_SUFFIX = ", first starts with an explanation of your judgement"
_SCORE_FENCE = "\n```"

FLUENCY_SYSTEM_PROMPT = (
    "You are a highly efficient assistant, who evaluates and selects the best "
    "large language model based on the quality of completion of a sentence. "
    "You will see a sentence to be completed and two completions from "
    "Assistant A and Assistant B and will have to decide which one is best. "
    "Make sure to not over-confidently prefer one assistant or the other and "
    "also make sure to not bias your preference based on the ordering or on "
    "the length of the answers."
)


@dataclass(frozen=True)
class JudgePromptPreset:
    name: str
    parse: JudgeParser | None = None
    """Parser for this preset's judge-output format; None only when delegated."""
    system_file: str | None = None
    user_file: str | None = None
    inline_system: str | None = None
    delegated: bool = False
    with_explanation: bool = False


@dataclass(frozen=True)
class ResolvedJudgePrompt:
    preset_name: str
    parse: JudgeParser | None
    system_prompt: str | None
    user_prompt_template: str
    source: PromptSource
    system_path: str | None = None
    user_path: str | None = None
    system_sha256: str | None = None
    user_sha256: str | None = None
    delegated: bool = False

    def metadata(self) -> dict[str, str | bool | None]:
        return {
            "judge_prompt_preset": self.preset_name,
            "judge_parser": parser_name(self.parse) if self.parse is not None else None,
            "judge_prompt_source": self.source,
            "judge_prompt_delegated": self.delegated,
            "judge_prompt_system_path": self.system_path,
            "judge_prompt_user_path": self.user_path,
            "judge_prompt_system_sha256": self.system_sha256,
            "judge_prompt_user_sha256": self.user_sha256,
        }


SCORE_PARSER = JUDGE_PARSERS["score"]

PRESETS: dict[str, JudgePromptPreset] = {
    DEFAULT_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=DEFAULT_JUDGE_PROMPT_PRESET,
        parse=SCORE_PARSER,
        system_file="system-prompt.txt",
        user_file="prompt.txt",
    ),
    DEFAULT_WITH_EXPLANATION_PRESET: JudgePromptPreset(
        name=DEFAULT_WITH_EXPLANATION_PRESET,
        parse=SCORE_PARSER,
        system_file="system-prompt.txt",
        user_file="prompt-with-explanation.txt",
        with_explanation=True,
    ),
    FLUENCY_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=FLUENCY_JUDGE_PROMPT_PRESET,
        parse=SCORE_PARSER,
        inline_system=FLUENCY_SYSTEM_PROMPT,
        user_file="prompt.txt",
    ),
    FASTCHAT_PAIRWISE_PROMPT_PRESET: JudgePromptPreset(
        name=FASTCHAT_PAIRWISE_PROMPT_PRESET,
        delegated=True,
    ),
    # Official Arena-Hard-Auto judge prompt (arena-hard-v0.1 judge_config.yaml),
    # verbatim except for placeholder names. The judge explains, then emits one
    # graded verdict label ([[A>>B]] ... [[B>>A]]).
    ARENA_HARD_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=ARENA_HARD_JUDGE_PROMPT_PRESET,
        parse=JUDGE_PARSERS["arena-hard-verdict"],
        system_file="arena-hard-system-prompt.txt",
        user_file="arena-hard-prompt.txt",
    ),
    # Official Arena-Hard v2.0 creative-writing variant: identical except the
    # judge is not asked to answer the prompt itself first.
    ARENA_HARD_CREATIVE_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=ARENA_HARD_CREATIVE_JUDGE_PROMPT_PRESET,
        parse=JUDGE_PARSERS["arena-hard-verdict"],
        system_file="arena-hard-creative-system-prompt.txt",
        user_file="arena-hard-prompt.txt",
    ),
}

JUDGE_PROMPT_PRESETS = tuple(PRESETS)


def default_preset_for_task(task: str | None) -> str:
    if task is None:
        return DEFAULT_JUDGE_PROMPT_PRESET
    # Import lazily: task validation uses the prompt catalog in this module.
    from judgearena.tasks.registry import get_packaged_task

    resolved = get_packaged_task(task)
    if resolved is not None and resolved.spec.protocol.judge.default_prompt:
        return resolved.spec.protocol.judge.default_prompt
    return DEFAULT_JUDGE_PROMPT_PRESET


def _task_file_prompt(
    task: str | None, *, multi_turn: bool
) -> ResolvedJudgePrompt | None:
    """Resolve a prompt shipped inside the task's folder, if declared."""
    if task is None:
        return None
    from judgearena.tasks.registry import get_packaged_task

    resolved = get_packaged_task(task)
    if resolved is None:
        return None
    prompt_spec = getattr(resolved.spec.protocol.judge, "prompt", None)
    if prompt_spec is None:
        return None
    system_prompt = resolved.prompt_texts[prompt_spec.system_file]
    user_prompt_template = _materialize_user_template(
        resolved.prompt_texts[prompt_spec.user_file],
        multi_turn=multi_turn,
        with_explanation=False,
    )
    return ResolvedJudgePrompt(
        preset_name=f"task:{resolved.definition_task}",
        parse=resolve_judge_parser(prompt_spec.parser),
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        source="task",
        system_path=prompt_spec.system_file,
        user_path=prompt_spec.user_file,
        system_sha256=_sha256(system_prompt),
        user_sha256=_sha256(user_prompt_template),
    )


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


def _resolve_file_prompt(
    *,
    system_file: str | Path,
    user_file: str | Path,
    multi_turn: bool,
    parse: JudgeParser,
) -> ResolvedJudgePrompt:
    system_path = Path(system_file)
    user_path = Path(user_file)
    system_prompt = system_path.read_text(encoding="utf-8")
    user_prompt_template = _materialize_user_template(
        user_path.read_text(encoding="utf-8"),
        multi_turn=multi_turn,
        with_explanation=False,
    )
    return ResolvedJudgePrompt(
        preset_name=f"file:{system_path.name}+{user_path.name}",
        parse=parse,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        source="file",
        system_path=str(system_path),
        user_path=str(user_path),
        system_sha256=_sha256(system_prompt),
        user_sha256=_sha256(user_prompt_template),
    )


def resolve_judge_prompt(
    *,
    task: str | None = None,
    preset: str | None = None,
    system_file: str | Path | None = None,
    user_file: str | Path | None = None,
    multi_turn: bool = False,
    parser: str | None = None,
) -> ResolvedJudgePrompt:
    if (system_file is None) != (user_file is None):
        raise ValueError(
            "Both --judge_system_prompt_file and --judge_user_prompt_file must "
            "be provided together."
        )
    if system_file is not None and user_file is not None:
        return _resolve_file_prompt(
            system_file=system_file,
            user_file=user_file,
            multi_turn=multi_turn,
            parse=resolve_judge_parser(parser or "score"),
        )
    if parser is not None:
        raise ValueError(
            "judge.parser requires judge prompt files; a preset already "
            "defines its parser."
        )

    if preset is None:
        task_prompt = _task_file_prompt(task, multi_turn=multi_turn)
        if task_prompt is not None:
            return task_prompt
        preset = default_preset_for_task(task)

    spec = PRESETS.get(preset)
    if spec is None:
        raise KeyError(
            f"Unknown judge prompt preset {preset!r}. Available: {sorted(PRESETS)}"
        )

    if spec.delegated:
        return ResolvedJudgePrompt(
            preset_name=spec.name,
            parse=spec.parse,
            system_prompt=None,
            user_prompt_template="",
            source="delegated",
            delegated=True,
        )

    if spec.user_file is None:
        raise ValueError(f"Judge prompt preset {spec.name!r} is missing a user file.")

    system_prompt = (
        spec.inline_system
        if spec.inline_system is not None
        else _load_packaged_text(spec.system_file)  # type: ignore[arg-type]
    )
    user_prompt_template = _materialize_user_template(
        _load_packaged_text(spec.user_file),
        multi_turn=multi_turn,
        with_explanation=spec.with_explanation,
    )
    return ResolvedJudgePrompt(
        preset_name=spec.name,
        parse=spec.parse,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        source="preset",
        system_path=spec.system_file,
        user_path=spec.user_file,
        system_sha256=_sha256(system_prompt),
        user_sha256=_sha256(user_prompt_template),
    )


def resolve_run_judge_prompt(
    task: str | None,
    judge_cfg,
    *,
    multi_turn: bool = False,
) -> ResolvedJudgePrompt:
    prompt = getattr(judge_cfg, "prompt", None)
    return resolve_judge_prompt(
        task=task,
        preset=getattr(judge_cfg, "prompt_preset", None),
        system_file=prompt.system_file if prompt is not None else None,
        user_file=prompt.user_file if prompt is not None else None,
        multi_turn=multi_turn,
        parser=prompt.parser if prompt is not None else None,
    )
