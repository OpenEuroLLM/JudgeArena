"""Named prompt templates and mode registry for meta-evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

from judgearena.meta_eval.cli_args import PROMPT_MODES
from judgearena.prompts.registry import resolve_judge_prompt


@dataclass(frozen=True)
class PromptModeSpec:
    name: str
    system_prompt: str | None = None
    user_prompt_template: str | None = None


def _read_prompt(filename: str) -> str:
    return (
        files("judgearena.meta_eval")
        .joinpath("prompts", filename)
        .read_text(encoding="utf-8")
    )


def resolve_prompt_mode(
    prompt_mode: str,
    *,
    provide_explanation: bool = False,
) -> PromptModeSpec:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt mode {prompt_mode!r}.")

    if prompt_mode == "standard":
        resolved = resolve_judge_prompt(
            provide_explanation=provide_explanation,
        )
        return PromptModeSpec(
            name=prompt_mode,
            system_prompt=resolved.system_prompt,
            user_prompt_template=resolved.user_prompt_template,
        )

    if prompt_mode == "arena-hard":
        return PromptModeSpec(
            name=prompt_mode,
            system_prompt=_read_prompt("arena_hard_system.txt"),
            user_prompt_template=_read_prompt("arena_hard_user.txt"),
        )

    if prompt_mode == "alpaca-eval":
        return PromptModeSpec(
            name=prompt_mode,
            system_prompt=_read_prompt("alpaca_eval_system.txt"),
            user_prompt_template=_read_prompt("alpaca_eval_user.txt"),
        )

    return PromptModeSpec(
        name=prompt_mode,
        system_prompt=_read_prompt("alpaca_eval_system.txt"),
        user_prompt_template=_read_prompt("alpaca_eval_pair_score_user.txt"),
    )
