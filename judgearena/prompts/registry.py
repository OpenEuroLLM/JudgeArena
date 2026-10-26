from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langchain_core.prompts import ChatPromptTemplate

from judgearena.prompts.criteria import prompt_block, resolve_criteria

if TYPE_CHECKING:
    from judgearena.tasks.schema import ResolvedTaskSpec

from judgearena.prompts.parsing import (
    JUDGE_PARSERS,
    CriteriaParser,
    JudgeParser,
    OutputParser,
    parser_name,
    resolve_judge_parser,
)

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
    parser: JudgeParser | None = None
    """Parser for this preset; None when delegated or configured during resolution."""
    system_file: str | None = None
    user_file: str | None = None
    inline_system: str | None = None
    delegated: bool = False
    with_explanation: bool = False
    output_format_file: str | None = None
    aggregation: Literal["individual", "agreement"] = "individual"
    score_renderer: Callable | None = None
    """Optional scalar-score renderer: (samples, prompt_preset, truncate_input_chars,
    strip_thinking_before_judging) -> (rendering fields, ParsedScore requests, prompts).
    Rows and requests follow the sample index; see judge_single_answers.
    """


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
    score_renderer: Callable | None = None

    def metadata(self) -> dict[str, str | bool | None]:
        return {
            "judge_prompt_preset": self.preset_name,
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


def build_pairwise_chat_prompt_template(
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


def _render_mt_bench_101_scores(samples, **kwargs):
    from judgearena.prompts.mt_bench_101 import render_mt_bench_101_requests

    return render_mt_bench_101_requests(samples, **kwargs)


SCORE_PARSER = JUDGE_PARSERS["score"]

PRESETS: dict[str, JudgePromptPreset] = {
    "criteria": JudgePromptPreset(
        name="criteria",
        system_file="system-prompt.txt",
        user_file="prompt.txt",
        output_format_file="prompt-criteria.txt",
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
    FLUENCY_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=FLUENCY_JUDGE_PROMPT_PRESET,
        parser=SCORE_PARSER,
        inline_system=FLUENCY_SYSTEM_PROMPT,
        user_file="prompt.txt",
        output_format_file="score-output.txt",
    ),
    FASTCHAT_PAIRWISE_PROMPT_PRESET: JudgePromptPreset(
        name=FASTCHAT_PAIRWISE_PROMPT_PRESET,
        delegated=True,
        aggregation="agreement",
    ),
    MT_BENCH_101_PROMPT_PRESET: JudgePromptPreset(
        name=MT_BENCH_101_PROMPT_PRESET,
        delegated=True,
        score_renderer=_render_mt_bench_101_scores,
    ),
    MT_BENCH_101_CLEAN_PROMPT_PRESET: JudgePromptPreset(
        name=MT_BENCH_101_CLEAN_PROMPT_PRESET,
        delegated=True,
        score_renderer=_render_mt_bench_101_scores,
    ),
    # Official Arena-Hard-Auto judge prompt (arena-hard-v0.1 judge_config.yaml),
    # verbatim except for placeholder names. The judge explains, then emits one
    # graded verdict label ([[A>>B]] ... [[B>>A]]).
    ARENA_HARD_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=ARENA_HARD_JUDGE_PROMPT_PRESET,
        parser=JUDGE_PARSERS["arena-hard-verdict"],
        system_file="arena-hard-system-prompt.txt",
        user_file="arena-hard-prompt.txt",
    ),
    # Official Arena-Hard v2.0 creative-writing variant: identical except the
    # judge is not asked to answer the prompt itself first.
    ARENA_HARD_CREATIVE_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=ARENA_HARD_CREATIVE_JUDGE_PROMPT_PRESET,
        parser=JUDGE_PARSERS["arena-hard-verdict"],
        system_file="arena-hard-creative-system-prompt.txt",
        user_file="arena-hard-prompt.txt",
    ),
    # Official AlpacaEval 2.0 annotator prompt (alpaca_eval_clf.txt), verbatim
    # except for placeholder names and brace escaping for f-string templating.
    # The judge answers with a single model identifier: "m" (completion_A) or
    # "M" (completion_B).
    ALPACA_EVAL_JUDGE_PROMPT_PRESET: JudgePromptPreset(
        name=ALPACA_EVAL_JUDGE_PROMPT_PRESET,
        parser=JUDGE_PARSERS["alpaca-eval-token"],
        system_file="alpaca-eval-system-prompt.txt",
        user_file="alpaca-eval-prompt.txt",
    ),
    META_EVAL_PAIR_SCORE_PROMPT_PRESET: JudgePromptPreset(
        name=META_EVAL_PAIR_SCORE_PROMPT_PRESET,
        parser=JUDGE_PARSERS["meta-eval-score"],
        system_file="system-prompt.txt",
        user_file="prompt.txt",
        output_format_file="integer-score-output.txt",
    ),
    META_EVAL_ALPACA_EVAL_JSON_PROMPT_PRESET: JudgePromptPreset(
        name=META_EVAL_ALPACA_EVAL_JSON_PROMPT_PRESET,
        parser=JUDGE_PARSERS["alpaca-eval-json"],
        system_file="meta-eval-alpaca-eval-system-prompt.txt",
        user_file="meta-eval-alpaca-eval-json-prompt.txt",
    ),
    META_EVAL_ALPACA_EVAL_PAIR_SCORE_PROMPT_PRESET: JudgePromptPreset(
        name=META_EVAL_ALPACA_EVAL_PAIR_SCORE_PROMPT_PRESET,
        parser=JUDGE_PARSERS["meta-eval-score"],
        system_file="meta-eval-alpaca-eval-system-prompt.txt",
        user_file="meta-eval-alpaca-eval-pair-score-prompt.txt",
    ),
}

JUDGE_PROMPT_PRESETS = tuple(PRESETS)


def default_preset_for_task(task: str | ResolvedTaskSpec | None) -> str:
    if task is None:
        return DEFAULT_JUDGE_PROMPT_PRESET
    if not isinstance(task, str):
        return task.spec.protocol.judge.default_prompt_preset
    # Import lazily: task validation uses the prompt catalog in this module.
    from judgearena.tasks.registry import get_packaged_task

    resolved = get_packaged_task(task)
    if resolved is not None:
        return resolved.spec.protocol.judge.default_prompt_preset
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
    task: str | ResolvedTaskSpec | None = None,
    preset: str | None = None,
    system_file: str | Path | None = None,
    user_file: str | Path | None = None,
    multi_turn: bool = False,
    parser: str | None = None,
    criteria_name: str | None = None,
    criteria_file: str | Path | None = None,
) -> ResolvedJudgePrompt:
    if preset is None:
        preset = default_preset_for_task(task)
    if (criteria_name is not None or criteria_file is not None) and (
        preset != "criteria" or system_file is not None or user_file is not None
    ):
        raise ValueError("criteria options require the 'criteria' prompt preset.")
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
            aggregation=spec.aggregation,
            score_renderer=spec.score_renderer,
        )

    if spec.user_file is None:
        raise ValueError(f"Judge prompt preset {spec.name!r} is missing a user file.")

    system_prompt = (
        spec.inline_system
        if spec.inline_system is not None
        else _load_packaged_text(spec.system_file)  # type: ignore[arg-type]
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
        _load_packaged_text(spec.user_file),
        multi_turn=multi_turn,
        with_explanation=spec.with_explanation,
    )
    resolved_parser = spec.parser
    if preset == "criteria":
        criteria = resolve_criteria(criteria_name or "default", criteria_file)
        names = [criterion.name for criterion in criteria]
        resolved_parser = CriteriaParser(names)
        fields = {
            "criteria_block": prompt_block(criteria),
            "criteria_output_lines": "\n".join(
                f"{name}: A=<score>, B=<score>" for name in names
            ),
        }
        # Substitute once: rubric text may contain literal braces or placeholder names.
        output_format = re.sub(
            r"\{(criteria_block|criteria_output_lines)\}",
            lambda match: fields[match[1]].replace("{", "{{").replace("}", "}}"),
            output_format,
        )
    if output_format is not None:
        user_prompt_template = (
            user_prompt_template.rstrip("\n") + "\n\n" + output_format
        )
    return ResolvedJudgePrompt(
        preset_name=spec.name,
        parser=resolved_parser,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        source="preset",
        system_path=spec.system_file,
        user_path=spec.user_file,
        system_sha256=_sha256(system_prompt),
        user_sha256=_sha256(user_prompt_template),
        output_format=output_format,
        aggregation=spec.aggregation,
        score_renderer=spec.score_renderer,
    )


def resolve_run_judge_prompt(
    task: str | ResolvedTaskSpec | None,
    judge_cfg,
    *,
    multi_turn: bool = False,
) -> ResolvedJudgePrompt:
    prompt = getattr(judge_cfg, "prompt", None)
    return resolve_judge_prompt(
        criteria_name=getattr(judge_cfg, "criteria_name", None),
        criteria_file=getattr(judge_cfg, "criteria_file", None),
        task=task,
        preset=getattr(judge_cfg, "prompt_preset", None),
        system_file=prompt.system_file if prompt is not None else None,
        user_file=prompt.user_file if prompt is not None else None,
        multi_turn=multi_turn,
        parser=prompt.parser if prompt is not None else None,
    )
