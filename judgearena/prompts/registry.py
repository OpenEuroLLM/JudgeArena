from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

JudgeParserMode = Literal["score", "criteria"]
PromptSource = Literal["preset", "file", "override", "delegated"]

DEFAULT_JUDGE_PROMPT_PRESET = "default"
DEFAULT_WITH_EXPLANATION_PRESET = "default_with_explanation"
FLUENCY_JUDGE_PROMPT_PRESET = "fluency"
FASTCHAT_PAIRWISE_PROMPT_PRESET = "fastchat-pairwise"
CRITERIA_JUDGE_PROMPT_PRESET = "criteria"

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
    parser_mode: JudgeParserMode = "score"
    system_file: str | None = None
    user_file: str | None = None
    inline_system: str | None = None
    delegated: bool = False
    with_explanation: bool = False
    criteria_name: str | None = None  # set => multi-criteria preset


@dataclass(frozen=True)
class ResolvedJudgePrompt:
    preset_name: str
    parser_mode: JudgeParserMode
    system_prompt: str | None
    user_prompt_template: str
    source: PromptSource
    system_path: str | None = None
    user_path: str | None = None
    system_sha256: str | None = None
    user_sha256: str | None = None
    delegated: bool = False
    criteria_names: tuple[str, ...] | None = None

    def metadata(self) -> dict[str, str | bool | None]:
        return {
            "judge_prompt_preset": self.preset_name,
            "judge_prompt_source": self.source,
            "judge_prompt_delegated": self.delegated,
            "judge_prompt_system_path": self.system_path,
            "judge_prompt_user_path": self.user_path,
            "judge_prompt_system_sha256": self.system_sha256,
            "judge_prompt_user_sha256": self.user_sha256,
        }


PRESETS: dict[str, JudgePromptPreset] = {
    DEFAULT_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=DEFAULT_JUDGE_PROMPT_PRESET,
        system_file="system-prompt.txt",
        user_file="prompt.txt",
    ),
    DEFAULT_WITH_EXPLANATION_PRESET: JudgePromptPreset(
        name=DEFAULT_WITH_EXPLANATION_PRESET,
        system_file="system-prompt.txt",
        user_file="prompt-with-explanation.txt",
        with_explanation=True,
    ),
    FLUENCY_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=FLUENCY_JUDGE_PROMPT_PRESET,
        inline_system=FLUENCY_SYSTEM_PROMPT,
        user_file="prompt.txt",
    ),
    FASTCHAT_PAIRWISE_PROMPT_PRESET: JudgePromptPreset(
        name=FASTCHAT_PAIRWISE_PROMPT_PRESET,
        delegated=True,
    ),
    CRITERIA_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=CRITERIA_JUDGE_PROMPT_PRESET,
        parser_mode="criteria",
        system_file="system-prompt.txt",
        user_file="prompt-criteria.txt",
        criteria_name="default",
    ),
}

JUDGE_PROMPT_PRESETS = tuple(PRESETS)

TASK_DEFAULT_PRESET: dict[str, str] = {
    "alpaca-eval": DEFAULT_JUDGE_PROMPT_PRESET,
    "arena-hard-v0.1": DEFAULT_JUDGE_PROMPT_PRESET,
    "arena-hard-v2.0": DEFAULT_JUDGE_PROMPT_PRESET,
    "mt-bench": FASTCHAT_PAIRWISE_PROMPT_PRESET,
}


def default_preset_for_task(task: str | None) -> str:
    if task is None:
        return DEFAULT_JUDGE_PROMPT_PRESET
    if task in TASK_DEFAULT_PRESET:
        return TASK_DEFAULT_PRESET[task]
    if task.startswith("m-arena-hard"):
        return DEFAULT_JUDGE_PROMPT_PRESET
    if task.startswith("fluency"):
        return FLUENCY_JUDGE_PROMPT_PRESET
    return DEFAULT_JUDGE_PROMPT_PRESET


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_packaged_text(filename: str) -> str:
    return files(PROMPTS_PACKAGE).joinpath(filename).read_text(encoding="utf-8")


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
    provide_explanation: bool,
) -> ResolvedJudgePrompt:
    system_path = Path(system_file)
    user_path = Path(user_file)
    system_prompt = system_path.read_text(encoding="utf-8")
    user_prompt_template = _materialize_user_template(
        user_path.read_text(encoding="utf-8"),
        multi_turn=multi_turn,
        with_explanation=provide_explanation,
    )
    return ResolvedJudgePrompt(
        preset_name=f"file:{system_path.name}+{user_path.name}",
        parser_mode="score",
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
    provide_explanation: bool = False,
    criteria_file: str | Path | None = None,
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
            provide_explanation=provide_explanation,
        )

    if preset is None:
        # `provide_explanation` is a legacy alias for explicitly selecting the
        # explanation preset. It intentionally takes precedence over the task
        # default (so e.g. `--provide_explanation` on any task yields
        # `default_with_explanation`); pass `--judge_prompt_preset` to opt out.
        preset = (
            DEFAULT_WITH_EXPLANATION_PRESET
            if provide_explanation
            else default_preset_for_task(task)
        )

    spec = PRESETS.get(preset)
    if spec is None:
        raise KeyError(
            f"Unknown judge prompt preset {preset!r}. Available: {sorted(PRESETS)}"
        )

    if criteria_file is not None and spec.criteria_name is None:
        raise ValueError(
            f"criteria_file is set but preset {spec.name!r} is not a criteria "
            "preset; set prompt_preset to a criteria preset (e.g. 'criteria')."
        )

    if spec.delegated:
        return ResolvedJudgePrompt(
            preset_name=spec.name,
            parser_mode=spec.parser_mode,
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

    criteria_names: tuple[str, ...] | None = None
    if spec.criteria_name is not None:
        from judgearena.criteria.io import resolve_criteria
        from judgearena.criteria.schema import criterion_names, prompt_block

        _, criteria = resolve_criteria(spec.criteria_name, criteria_file)
        names = criterion_names(criteria)
        criteria_names = tuple(names)
        output_lines = "\n".join(f"{n}: A=<1-5> B=<1-5>" for n in names)
        user_prompt_template = user_prompt_template.replace(
            "{criteria_block}", prompt_block(criteria)
        ).replace("{criteria_output_lines}", output_lines)

    return ResolvedJudgePrompt(
        preset_name=spec.name,
        parser_mode=spec.parser_mode,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        source="preset",
        system_path=spec.system_file,
        user_path=spec.user_file,
        system_sha256=_sha256(system_prompt),
        user_sha256=_sha256(user_prompt_template),
        criteria_names=criteria_names,
    )


def resolve_run_judge_prompt(
    task: str | None,
    judge_cfg,
    *,
    multi_turn: bool = False,
) -> ResolvedJudgePrompt:
    return resolve_judge_prompt(
        task=task,
        preset=getattr(judge_cfg, "prompt_preset", None),
        system_file=getattr(judge_cfg, "system_prompt_file", None),
        user_file=getattr(judge_cfg, "user_prompt_file", None),
        multi_turn=multi_turn,
        provide_explanation=getattr(judge_cfg, "provide_explanation", False),
        criteria_file=getattr(judge_cfg, "criteria_file", None),
    )
