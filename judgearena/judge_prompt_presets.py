from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

JudgeParserMode = Literal["score", "verdict"]

DEFAULT_JUDGE_PROMPT_PRESET = "default"
SKYWORK_JUDGE_PROMPT_PRESET = "skywork"
JUDGE_PROMPT_PRESETS = (
    DEFAULT_JUDGE_PROMPT_PRESET,
    SKYWORK_JUDGE_PROMPT_PRESET,
)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_EXPLANATION_SUFFIX = ", first starts with an explanation of your judgement"
_SCORE_FENCE = "\n```"


@dataclass(frozen=True)
class PairwiseJudgePromptPreset:
    name: str
    parser_mode: JudgeParserMode
    system_prompt_filename: str | None
    user_prompt_filename: str
    user_prompt_with_explanation_filename: str


@dataclass(frozen=True)
class ResolvedJudgePrompt:
    preset_name: str
    parser_mode: JudgeParserMode
    system_prompt: str | None
    user_prompt_template: str


_PAIRWISE_PROMPT_PRESETS: dict[str, PairwiseJudgePromptPreset] = {
    DEFAULT_JUDGE_PROMPT_PRESET: PairwiseJudgePromptPreset(
        name=DEFAULT_JUDGE_PROMPT_PRESET,
        parser_mode="score",
        system_prompt_filename="system-prompt.txt",
        user_prompt_filename="prompt.txt",
        user_prompt_with_explanation_filename="prompt-with-explanation.txt",
    ),
    SKYWORK_JUDGE_PROMPT_PRESET: PairwiseJudgePromptPreset(
        name=SKYWORK_JUDGE_PROMPT_PRESET,
        parser_mode="verdict",
        system_prompt_filename=None,
        user_prompt_filename="skywork-prompt.txt",
        user_prompt_with_explanation_filename="skywork-prompt-with-explanation.txt",
    ),
}


def _render_user_prompt_template(
    raw_template: str, *, provide_explanation: bool
) -> str:
    template = raw_template.replace(
        "{explanation_suffix}",
        _EXPLANATION_SUFFIX if provide_explanation else _SCORE_FENCE,
    )
    return template


def resolve_pairwise_judge_prompt(
    *,
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
    provide_explanation: bool,
    multi_turn: bool = False,
    system_prompt: str | None = None,
    user_prompt_template: str | None = None,
) -> ResolvedJudgePrompt:
    preset = _PAIRWISE_PROMPT_PRESETS.get(prompt_preset)
    if preset is None:
        supported = ", ".join(sorted(_PAIRWISE_PROMPT_PRESETS))
        raise ValueError(
            f"Unsupported judge prompt preset '{prompt_preset}'. Choose from: {supported}."
        )

    prompt_filename = (
        preset.user_prompt_with_explanation_filename
        if provide_explanation
        else preset.user_prompt_filename
    )
    default_system_prompt = (
        (_PROMPTS_DIR / preset.system_prompt_filename).read_text(encoding="utf-8")
        if preset.system_prompt_filename is not None
        else None
    )
    default_user_prompt_template = _render_user_prompt_template(
        (_PROMPTS_DIR / prompt_filename).read_text(encoding="utf-8"),
        provide_explanation=provide_explanation,
    )
    return ResolvedJudgePrompt(
        preset_name=preset.name,
        parser_mode=preset.parser_mode,
        system_prompt=system_prompt
        if system_prompt is not None
        else default_system_prompt,
        user_prompt_template=user_prompt_template
        if user_prompt_template is not None
        else default_user_prompt_template,
    )
