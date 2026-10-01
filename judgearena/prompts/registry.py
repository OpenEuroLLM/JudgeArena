from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from judgearena.prompts.parsing import (
    JUDGE_PARSERS,
    JudgeParser,
    parser_name,
    resolve_judge_parser,
)
from judgearena.prompts.presets import PROMPT_PRESETS

PromptSource = Literal["preset", "file", "override", "delegated"]

DEFAULT_JUDGE_PROMPT_PRESET = "default"
DEFAULT_WITH_EXPLANATION_PRESET = "default_with_explanation"
FLUENCY_JUDGE_PROMPT_PRESET = "fluency"
FASTCHAT_PAIRWISE_PROMPT_PRESET = "fastchat-pairwise"
MT_BENCH_101_PROMPT_PRESET = "mt-bench-101"
MT_BENCH_101_CLEAN_PROMPT_PRESET = "mt-bench-101-clean"
ARENA_HARD_JUDGE_PROMPT_PRESET = "arena-hard"
ARENA_HARD_CREATIVE_JUDGE_PROMPT_PRESET = "arena-hard-creative"
ALPACA_EVAL_JUDGE_PROMPT_PRESET = "alpaca-eval"
META_EVAL_PAIR_SCORE_PROMPT_PRESET = "meta-eval-pair-score"
META_EVAL_ALPACA_EVAL_JSON_PROMPT_PRESET = "meta-eval-alpaca-eval-json"
META_EVAL_ALPACA_EVAL_PAIR_SCORE_PROMPT_PRESET = "meta-eval-alpaca-eval-pair-score"

_COMPLETION_LABEL_SINGLE = "Answer"
_COMPLETION_LABEL_MULTI_TURN = "Conversation with User"
_EXPLANATION_SUFFIX = ", first starts with an explanation of your judgement"
_SCORE_FENCE = "\n```"


@dataclass(frozen=True)
class JudgePromptPreset:
    name: str
    source_path: str | None = None
    parser: JudgeParser | None = None
    """Parser for this preset's judge-output format; None only when delegated."""
    system_prompt: str | None = None
    user_prompt_template: str | None = None
    delegated: bool = False
    with_explanation: bool = False


@dataclass(frozen=True)
class ResolvedJudgePrompt:
    preset_name: str
    parser: JudgeParser | None
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
            "judge_parser": (
                parser_name(self.parser) if self.parser is not None else None
            ),
            "judge_prompt_source": self.source,
            "judge_prompt_delegated": self.delegated,
            "judge_prompt_system_path": self.system_path,
            "judge_prompt_user_path": self.user_path,
            "judge_prompt_system_sha256": self.system_sha256,
            "judge_prompt_user_sha256": self.user_sha256,
        }


PRESETS: dict[str, JudgePromptPreset] = {
    name: JudgePromptPreset(
        name=name,
        source_path=preset.source_path,
        parser=(JUDGE_PARSERS[preset.parser] if preset.parser is not None else None),
        system_prompt=preset.system_prompt,
        user_prompt_template=preset.user_prompt_template,
        delegated=preset.delegated,
        with_explanation=preset.with_explanation,
    )
    for name, preset in PROMPT_PRESETS.items()
}

JUDGE_PROMPT_PRESETS = tuple(PRESETS)


def default_preset_for_task(task: str | None) -> str:
    if task is None:
        return DEFAULT_JUDGE_PROMPT_PRESET
    # Import lazily: task validation uses the prompt catalog in this module.
    from judgearena.tasks.registry import get_packaged_task

    resolved = get_packaged_task(task)
    if resolved is not None:
        return resolved.spec.protocol.judge.default_prompt_preset
    return DEFAULT_JUDGE_PROMPT_PRESET


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    parser: JudgeParser,
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
        parser=parser,
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
            parser=resolve_judge_parser(parser or "score"),
        )
    if parser is not None:
        raise ValueError(
            "judge.parser requires judge prompt files; a preset already "
            "defines its parser."
        )

    if preset is None:
        preset = default_preset_for_task(task)

    spec = PRESETS.get(preset)
    if spec is None:
        raise KeyError(
            f"Unknown judge prompt preset {preset!r}. Available: {sorted(PRESETS)}"
        )

    if spec.delegated:
        return ResolvedJudgePrompt(
            preset_name=spec.name,
            parser=spec.parser,
            system_prompt=None,
            user_prompt_template="",
            source="delegated",
            delegated=True,
        )

    if spec.system_prompt is None or spec.user_prompt_template is None:
        raise ValueError(f"Judge prompt preset {spec.name!r} is missing prompt text.")

    system_prompt = spec.system_prompt
    user_prompt_template = _materialize_user_template(
        spec.user_prompt_template,
        multi_turn=multi_turn,
        with_explanation=spec.with_explanation,
    )
    return ResolvedJudgePrompt(
        preset_name=spec.name,
        parser=spec.parser,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        source="preset",
        system_path=spec.source_path,
        user_path=spec.source_path,
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
