"""Judge prompt presets and per-task default mapping.

JudgeArena ships a small registry of named prompt presets so that every
benchmark gets a sensible default that is *also* recorded by hash in the
run metadata.  Users can either pick a preset by name with
``--judge_prompt_preset NAME`` or supply a custom ``(system, user)`` pair
with ``--judge_system_prompt_file`` / ``--judge_user_prompt_file``.

The MT-Bench pipeline keeps its own category-aware prompt selection (see
:mod:`judgearena.mt_bench.fastchat_compat`); the registry just records the
preset name ``fastchat-pairwise`` so the metadata bundle still answers
"which judge prompt was used here?".
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

PROMPTS_PACKAGE = "judgearena.prompts"

FLUENCY_SYSTEM = (
    "You are a highly efficient assistant, who evaluates and selects the best "
    "large language model based on the quality of completion of a sentence. "
    "You will see a sentence to be completed and two completions from "
    "Assistant A and Assistant B and will have to decide which one is best. "
    "Make sure to not over-confidently prefer one assistant or the other and "
    "also make sure to not bias your preference based on the ordering or on "
    "the length of the answers."
)


@dataclass(frozen=True)
class _PresetSpec:
    """Internal description of a prompt preset.

    Either ``system_file`` or ``inline_system`` is set (file wins when both).
    ``user_file`` is required for non-delegated presets.  ``delegated``
    presets do not produce ``(system, user)`` strings; callers must use
    their own prompt-selection machinery.
    """

    system_file: str | None = None
    inline_system: str | None = None
    user_file: str | None = None
    delegated: bool = False


PRESETS: dict[str, _PresetSpec] = {
    "default": _PresetSpec(system_file="system-prompt.txt", user_file="prompt.txt"),
    "default_with_explanation": _PresetSpec(
        system_file="system-prompt.txt", user_file="prompt-with-explanation.txt"
    ),
    "fluency": _PresetSpec(inline_system=FLUENCY_SYSTEM, user_file="prompt.txt"),
    "fastchat-pairwise": _PresetSpec(delegated=True),
}


# Per-task default preset.  Tasks not listed fall back through prefix rules
# in ``default_preset_for_task`` (m-arena-hard*, fluency-*) and ultimately
# to ``default``.
TASK_DEFAULT_PRESET: dict[str, str] = {
    "alpaca-eval": "default",
    "arena-hard-v0.1": "default",
    "arena-hard-v2.0": "default",
    "m-arena-hard": "default",
    "mt-bench": "fastchat-pairwise",
}


@dataclass(frozen=True)
class ResolvedJudgePrompt:
    """The judge prompt that will actually be used for a run.

    ``delegated=True`` signals that the calling pipeline (currently only
    MT-Bench) selects its templates per item; ``system_text`` and
    ``user_template_text`` are then unused but the ``name`` is still
    recorded in the run metadata.
    """

    name: str
    system_text: str
    user_template_text: str
    delegated: bool = False
    source: str = "preset"  # "preset" or "file"
    system_path: str | None = None
    user_path: str | None = None


_COMPLETION_LABEL_SINGLE = "Answer"
_COMPLETION_LABEL_MULTI_TURN = "Conversation with User"
_EXPLANATION_SUFFIX = ", first starts with an explanation of your judgement"
_SCORE_FENCE = "\n```"


def default_preset_for_task(task: str | None) -> str:
    """Return the preset name that should be used for ``task`` by default."""
    if not task:
        return "default"
    if task in TASK_DEFAULT_PRESET:
        return TASK_DEFAULT_PRESET[task]
    if task.startswith("m-arena-hard"):
        return "default"
    if task.startswith("fluency-") or task.startswith("fluency"):
        return "fluency"
    return "default"


def _load_packaged_text(filename: str) -> str:
    return files(PROMPTS_PACKAGE).joinpath(filename).read_text(encoding="utf-8")


def _materialize_user_template(
    text: str, *, multi_turn: bool, with_explanation: bool
) -> str:
    """Apply the ``{completion_label}`` and ``{explanation_suffix}`` substitutions.

    These placeholders exist in the bundled ``prompt.txt`` but not in
    ``prompt-with-explanation.txt``; both spellings are handled idempotently
    so callers don't need to know which preset they got.
    """
    text = text.replace(
        "{completion_label}",
        _COMPLETION_LABEL_MULTI_TURN if multi_turn else _COMPLETION_LABEL_SINGLE,
    )
    text = text.replace(
        "{explanation_suffix}",
        _EXPLANATION_SUFFIX if with_explanation else _SCORE_FENCE,
    )
    return text


def resolve_judge_prompt(
    *,
    task: str | None = None,
    preset: str | None = None,
    system_file: str | Path | None = None,
    user_file: str | Path | None = None,
    multi_turn: bool = False,
    provide_explanation: bool = False,
) -> ResolvedJudgePrompt:
    """Resolve the prompt that should be used for this run.

    Resolution order:

    1. If both ``system_file`` and ``user_file`` are given, they win.
    2. Else if ``preset`` is given, it is used.
    3. Else if ``provide_explanation=True``, ``default_with_explanation`` is used
       (legacy alias kept for backward compatibility).
    4. Else the per-task default preset is selected (see
       :data:`TASK_DEFAULT_PRESET`).
    """
    if (system_file is None) != (user_file is None):
        raise ValueError(
            "Both --judge_system_prompt_file and --judge_user_prompt_file "
            "must be provided together."
        )

    if system_file is not None:
        sys_path = Path(system_file)
        usr_path = Path(user_file)  # type: ignore[arg-type]
        sys_text = sys_path.read_text(encoding="utf-8")
        usr_text = _materialize_user_template(
            usr_path.read_text(encoding="utf-8"),
            multi_turn=multi_turn,
            with_explanation=provide_explanation,
        )
        return ResolvedJudgePrompt(
            name=f"file:{sys_path.name}+{usr_path.name}",
            system_text=sys_text,
            user_template_text=usr_text,
            source="file",
            system_path=str(sys_path),
            user_path=str(usr_path),
        )

    if preset is None:
        if provide_explanation:
            preset = "default_with_explanation"
        else:
            preset = default_preset_for_task(task)

    if preset not in PRESETS:
        raise KeyError(
            f"Unknown judge prompt preset {preset!r}. Available: {sorted(PRESETS)}"
        )
    spec = PRESETS[preset]
    if spec.delegated:
        return ResolvedJudgePrompt(
            name=preset,
            system_text="",
            user_template_text="",
            delegated=True,
            source="preset",
        )

    sys_text = (
        spec.inline_system
        if spec.inline_system is not None
        else _load_packaged_text(spec.system_file)  # type: ignore[arg-type]
    )
    user_text = _load_packaged_text(spec.user_file)  # type: ignore[arg-type]
    user_text = _materialize_user_template(
        user_text,
        multi_turn=multi_turn,
        with_explanation=(preset == "default_with_explanation"),
    )
    return ResolvedJudgePrompt(
        name=preset,
        system_text=sys_text,
        user_template_text=user_text,
        source="preset",
        system_path=spec.system_file,
        user_path=spec.user_file,
    )
