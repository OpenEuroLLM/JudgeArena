from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

JudgeParserMode = Literal["score", "verdict"]

DEFAULT_JUDGE_PROMPT_PRESET = "default"
SKYWORK_JUDGE_PROMPT_PRESET = "skywork"

# m-ArenaHard v2.0 itself spans many more languages; these are only the
# localized prompt variants that we currently ship translations for.
M_ARENA_HARD_V2_LOCALIZED_PROMPT_PRESETS = {
    "ar": "m-arena-hard-v2-localized-ar",
    "pl": "m-arena-hard-v2-localized-pl",
    "uk": "m-arena-hard-v2-localized-uk",
    "zh": "m-arena-hard-v2-localized-zh",
}
JUDGE_PROMPT_PRESETS = (
    DEFAULT_JUDGE_PROMPT_PRESET,
    SKYWORK_JUDGE_PROMPT_PRESET,
    *M_ARENA_HARD_V2_LOCALIZED_PROMPT_PRESETS.values(),
)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_COMPLETION_LABEL_SINGLE = "Answer"
_COMPLETION_LABEL_MULTI_TURN = "Conversation with User"
_EXPLANATION_SUFFIX = ", first starts with an explanation of your judgement"
_SCORE_FENCE = "\n```"


@dataclass(frozen=True)
class PairwiseJudgePromptPreset:
    name: str
    parser_mode: JudgeParserMode
    system_prompt_filename: str | None
    user_prompt_filename: str
    user_prompt_with_explanation_filename: str
    supports_explanation: bool = True
    completion_label_single: str = _COMPLETION_LABEL_SINGLE
    completion_label_multi_turn: str = _COMPLETION_LABEL_MULTI_TURN


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
    M_ARENA_HARD_V2_LOCALIZED_PROMPT_PRESETS["ar"]: PairwiseJudgePromptPreset(
        name=M_ARENA_HARD_V2_LOCALIZED_PROMPT_PRESETS["ar"],
        parser_mode="score",
        system_prompt_filename="m_arena_hard_v2_localized/ar/system-prompt.txt",
        user_prompt_filename="m_arena_hard_v2_localized/ar/prompt.txt",
        user_prompt_with_explanation_filename="m_arena_hard_v2_localized/ar/prompt.txt",
        supports_explanation=False,
        completion_label_single="إجابة",
        completion_label_multi_turn="محادثة مع المستخدم",
    ),
    M_ARENA_HARD_V2_LOCALIZED_PROMPT_PRESETS["pl"]: PairwiseJudgePromptPreset(
        name=M_ARENA_HARD_V2_LOCALIZED_PROMPT_PRESETS["pl"],
        parser_mode="score",
        system_prompt_filename="m_arena_hard_v2_localized/pl/system-prompt.txt",
        user_prompt_filename="m_arena_hard_v2_localized/pl/prompt.txt",
        user_prompt_with_explanation_filename="m_arena_hard_v2_localized/pl/prompt.txt",
        supports_explanation=False,
        completion_label_single="odpowiedzi",
        completion_label_multi_turn="rozmowy z użytkownikiem",
    ),
    M_ARENA_HARD_V2_LOCALIZED_PROMPT_PRESETS["uk"]: PairwiseJudgePromptPreset(
        name=M_ARENA_HARD_V2_LOCALIZED_PROMPT_PRESETS["uk"],
        parser_mode="score",
        system_prompt_filename="m_arena_hard_v2_localized/uk/system-prompt.txt",
        user_prompt_filename="m_arena_hard_v2_localized/uk/prompt.txt",
        user_prompt_with_explanation_filename="m_arena_hard_v2_localized/uk/prompt.txt",
        supports_explanation=False,
        completion_label_single="відповіді",
        completion_label_multi_turn="розмови з користувачем",
    ),
    M_ARENA_HARD_V2_LOCALIZED_PROMPT_PRESETS["zh"]: PairwiseJudgePromptPreset(
        name=M_ARENA_HARD_V2_LOCALIZED_PROMPT_PRESETS["zh"],
        parser_mode="score",
        system_prompt_filename="m_arena_hard_v2_localized/zh/system-prompt.txt",
        user_prompt_filename="m_arena_hard_v2_localized/zh/prompt.txt",
        user_prompt_with_explanation_filename="m_arena_hard_v2_localized/zh/prompt.txt",
        supports_explanation=False,
        completion_label_single="回答",
        completion_label_multi_turn="与用户的对话",
    ),
}


def _render_user_prompt_template(
    raw_template: str,
    *,
    provide_explanation: bool,
    multi_turn: bool,
    completion_label_single: str = _COMPLETION_LABEL_SINGLE,
    completion_label_multi_turn: str = _COMPLETION_LABEL_MULTI_TURN,
) -> str:
    template = raw_template.replace(
        "{completion_label}",
        completion_label_multi_turn if multi_turn else completion_label_single,
    )
    template = template.replace(
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
    if provide_explanation and not preset.supports_explanation:
        raise ValueError(
            f"Judge prompt preset '{prompt_preset}' does not support "
            "provide_explanation=True."
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
        multi_turn=multi_turn,
        completion_label_single=preset.completion_label_single,
        completion_label_multi_turn=preset.completion_label_multi_turn,
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
