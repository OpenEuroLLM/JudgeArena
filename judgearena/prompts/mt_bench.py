"""MT-Bench prompt selection and conversation rendering."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from functools import partial
from importlib.resources import files
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import Field, model_validator

from judgearena.judging import JudgeRequest
from judgearena.prompts.parsing import FastChatVerdictParser
from judgearena.prompts.registry import (
    DEFAULT_JUDGE_PROMPT_PRESET,
    FASTCHAT_PAIRWISE_PROMPT_PRESET,
    ResolvedJudgePrompt,
    build_pairwise_chat_prompt_template,
    resolve_run_judge_prompt,
)
from judgearena.tasks.schema.base import StrictFrozenModel
from judgearena.utils import strip_thinking_tags

if TYPE_CHECKING:
    from judgearena.config import RunConfig

_PROMPT_PACKAGE = "judgearena.prompts"
_USER_SINGLE_BASE_FILE = "user-single-base.txt"
_USER_MULTI_BASE_FILE = "user-multi-base.txt"
_USER_SINGLE_REF_BLOCK_FILE = "user-single-reference-block.txt"
_USER_MULTI_REF_BLOCK_FILE = "user-multi-reference-block.txt"
_SYSTEM_BASE_FILE = "system-base.txt"


class MTBenchPromptOptions(StrictFrozenModel):
    """FastChat prompt variants and reference routing."""

    fastchat_prompt_preset: str = Field(default="default", min_length=1)
    fastchat_temperature: float = Field(default=0.0, ge=0)
    reference_categories: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_reference_categories(self):
        if any(not category for category in self.reference_categories):
            raise ValueError("reference categories must not be empty")
        if len(set(self.reference_categories)) != len(self.reference_categories):
            raise ValueError("reference categories must not contain duplicates")
        return self


def load_mt_bench_prompt_text(filename: str) -> str:
    return (
        files(_PROMPT_PACKAGE)
        .joinpath("templates", "mt_bench", filename)
        .read_text(encoding="utf-8")
    )


def render_mt_bench_prompt_text(filename: str, **kwargs: str) -> str:
    return load_mt_bench_prompt_text(filename).format(**kwargs)


def build_mt_bench_user_prompt_template(*, multi_turn: bool, ref_based: bool) -> str:
    base_filename = _USER_MULTI_BASE_FILE if multi_turn else _USER_SINGLE_BASE_FILE
    reference_block = ""
    if ref_based:
        ref_block_filename = (
            _USER_MULTI_REF_BLOCK_FILE if multi_turn else _USER_SINGLE_REF_BLOCK_FILE
        )
        reference_block = (
            load_mt_bench_prompt_text(ref_block_filename).rstrip("\n") + "\n\n"
        )
    return render_mt_bench_prompt_text(base_filename, reference_block=reference_block)


def _build_system_prompt(
    *,
    user_subject: str,
    task_description: str,
    begin_instruction: str,
    focus_line: str = "",
) -> str:
    focus_segment = f"{focus_line} " if focus_line else ""
    return render_mt_bench_prompt_text(
        _SYSTEM_BASE_FILE,
        user_subject=user_subject,
        task_description=task_description,
        focus_line=focus_segment,
        begin_instruction=begin_instruction,
    )


def _load_pairwise_prompt(
    *,
    name: str,
    multi_turn: bool,
    ref_based: bool,
    system_user_subject: str,
    system_task_description: str,
    system_begin_instruction: str,
    system_focus_line: str = "",
) -> ResolvedJudgePrompt:
    system = _build_system_prompt(
        user_subject=system_user_subject,
        task_description=system_task_description,
        begin_instruction=system_begin_instruction,
        focus_line=system_focus_line,
    )
    user = build_mt_bench_user_prompt_template(
        multi_turn=multi_turn, ref_based=ref_based
    )
    return ResolvedJudgePrompt(
        preset_name=FASTCHAT_PAIRWISE_PROMPT_PRESET,
        variant_name=name,
        parser=FastChatVerdictParser(),
        source="preset",
        system_prompt=system,
        user_prompt_template=user,
        system_sha256=hashlib.sha256(system.encode("utf-8")).hexdigest(),
        user_sha256=hashlib.sha256(user.encode("utf-8")).hexdigest(),
        aggregation="agreement",
    )


_PAIR_V2 = _load_pairwise_prompt(
    name="pair-v2",
    multi_turn=False,
    ref_based=False,
    system_user_subject="question displayed below",
    system_task_description=(
        "You should choose the assistant that follows the user's instructions and answers "
        "the user's question better. Your evaluation should consider factors such as the "
        "helpfulness, relevance, accuracy, depth, creativity, and level of detail of their "
        "responses."
    ),
    system_begin_instruction="comparing the two responses and provide a short explanation",
)

_PAIR_V2_MULTI = _load_pairwise_prompt(
    name="pair-v2-multi-turn",
    multi_turn=True,
    ref_based=False,
    system_user_subject="questions",
    system_task_description=(
        "You should choose the assistant that follows the user's instructions and answers "
        "the user's questions better. Your evaluation should consider factors such as the "
        "helpfulness, relevance, accuracy, depth, creativity, and level of detail of their "
        "responses."
    ),
    system_focus_line="You should focus on who provides a better answer to the second user question.",
    system_begin_instruction=(
        "comparing the responses of the two assistants and provide a short explanation"
    ),
)

_PAIR_MATH_V1 = _load_pairwise_prompt(
    name="pair-math-v1",
    multi_turn=False,
    ref_based=True,
    system_user_subject="question displayed below",
    system_task_description=(
        "Your evaluation should consider correctness and helpfulness. You will be given a "
        "reference answer, assistant A's answer, and assistant B's answer. Your job is to "
        "evaluate which assistant's answer is better."
    ),
    system_begin_instruction=(
        "comparing both assistants' answers with the reference answer. Identify and correct any mistakes"
    ),
)

_PAIR_MATH_V1_MULTI = _load_pairwise_prompt(
    name="pair-math-v1-multi-turn",
    multi_turn=True,
    ref_based=True,
    system_user_subject="questions",
    system_task_description=(
        "Your evaluation should consider correctness and helpfulness. You will be given "
        "reference answers, the assistant A's answers, the assistant B's answers. Your job is "
        "to determine which assistant provides correct and helpful answers to the second user question."
    ),
    system_begin_instruction=(
        "comparing both assistants' answers with the reference answers. Identify and correct any mistakes"
    ),
)


_FASTCHAT_PROMPT_PRESET_REGISTRY: dict[str, dict[str, ResolvedJudgePrompt]] = {
    DEFAULT_JUDGE_PROMPT_PRESET: {
        "single": _PAIR_V2,
        "multi": _PAIR_V2_MULTI,
        "single_ref": _PAIR_MATH_V1,
        "multi_ref": _PAIR_MATH_V1_MULTI,
    },
}


def _select_prompt(
    category: str | None,
    multi_turn: bool,
    *,
    reference_categories: Collection[str],
    prompt_preset: str = DEFAULT_JUDGE_PROMPT_PRESET,
) -> ResolvedJudgePrompt:
    prompt_variants = _FASTCHAT_PROMPT_PRESET_REGISTRY.get(prompt_preset)
    if prompt_variants is None:
        supported = ", ".join(sorted(_FASTCHAT_PROMPT_PRESET_REGISTRY))
        raise ValueError(
            f"Unsupported MT-Bench prompt preset '{prompt_preset}'. Choose from: {supported}."
        )
    needs_ref = (category or "") in reference_categories
    if needs_ref and multi_turn:
        return prompt_variants["multi_ref"]
    if needs_ref:
        return prompt_variants["single_ref"]
    if multi_turn:
        return prompt_variants["multi"]
    return prompt_variants["single"]


def _build_mt_bench_preset_user_prompt_template(
    *,
    resolved_prompt: ResolvedJudgePrompt,
    multi_turn: bool,
    ref_based: bool,
) -> str:
    if resolved_prompt.output_format is None:
        return resolved_prompt.user_prompt_template
    base_template = build_mt_bench_user_prompt_template(
        multi_turn=multi_turn,
        ref_based=ref_based,
    )
    return f"{base_template}\n\n{resolved_prompt.output_format}"


def build_mt_bench_preset_prompts(
    resolved_by_turn: dict[bool, ResolvedJudgePrompt],
) -> dict[tuple[bool, bool], ResolvedJudgePrompt]:
    """Prepare variants keyed by (multi_turn, ref_based) before model work."""
    prompts = {}
    for multi_turn, resolved in resolved_by_turn.items():
        if resolved.delegated:
            raise ValueError(
                f"Judge prompt preset '{resolved.preset_name}' is delegated and "
                "cannot be used for MT-Bench preset judging."
            )
        for ref_based in (False, True):
            suffix = "multi" if multi_turn else "single"
            if ref_based:
                suffix += "_ref"
            template = _build_mt_bench_preset_user_prompt_template(
                resolved_prompt=resolved,
                multi_turn=multi_turn,
                ref_based=ref_based,
            )
            prompts[multi_turn, ref_based] = replace(
                resolved,
                variant_name=f"{resolved.preset_name}-{suffix}",
                user_prompt_template=template,
                user_sha256=hashlib.sha256(template.encode("utf-8")).hexdigest(),
            )
    return prompts


@dataclass(frozen=True)
class MTBenchJudgeItem:
    id: str
    prompt: ResolvedJudgePrompt
    prompt_kwargs: dict[str, str]

    @property
    def prompt_name(self) -> str:
        return self.prompt.variant_name or self.prompt.preset_name


def group_indices_by_prompt_key(items: list[MTBenchJudgeItem]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        grouped.setdefault(item.prompt_name, []).append(idx)
    return grouped


def render_mt_bench_requests(comparisons: pd.DataFrame, **kwargs):
    """Prepare prompt batches; shared execution owns ordering and cache identity."""
    items = build_mt_bench_items(comparisons, **kwargs)
    for indices in group_indices_by_prompt_key(items).values():
        group = [items[i] for i in indices]
        inputs = build_pairwise_chat_prompt_template(group[0].prompt).batch(
            [item.prompt_kwargs for item in group]
        )
        yield [
            JudgeRequest(item.id, value, item.prompt.parser)
            for item, value in zip(group, inputs, strict=True)
        ]


def build_mt_bench_items(
    comparisons: pd.DataFrame,
    *,
    select_prompt: Callable[[str | None, bool], ResolvedJudgePrompt],
    truncate_input_chars: int | None = None,
    strip_thinking_before_judging: bool = False,
) -> list[MTBenchJudgeItem]:
    """Render the benchmark's one/two-turn variables from candidate conversations."""
    from judgearena.datasets.data import validate_messages
    from judgearena.utils import safe_text

    items = []
    for identity, row in comparisons.iterrows():
        a, b = row.messages_a, row.messages_b
        validate_messages(a)
        validate_messages(b)
        turn = row.turn
        if turn not in (1, 2) or len(a) != turn * 2 or len(b) != turn * 2:
            raise ValueError(
                "MT-Bench prompts require one or two user/assistant turns."
            )
        if any(a[i]["content"] != b[i]["content"] for i in range(0, len(a), 2)):
            raise ValueError(
                "MT-Bench prompts require matching user questions across candidates."
            )
        prompt = select_prompt(row.get("category"), turn == 2)

        def text(value, *, answer=False):
            value = safe_text(value, truncate_input_chars)
            return (
                strip_thinking_tags(value)
                if answer and strip_thinking_before_judging
                else value
            )

        kwargs = {}
        for i in range(turn):
            suffix = f"_{i + 1}" if turn == 2 else ""
            kwargs[f"question{suffix}"] = text(a[2 * i]["content"])
            kwargs[f"answer_a{suffix}"] = text(a[2 * i + 1]["content"], answer=True)
            kwargs[f"answer_b{suffix}"] = text(b[2 * i + 1]["content"], answer=True)
            if f"{{ref_answer_{i + 1}}}" in prompt.user_prompt_template:
                kwargs[f"ref_answer_{i + 1}"] = text(row.get(f"reference_turn_{i + 1}"))
        items.append(MTBenchJudgeItem(str(identity), prompt, kwargs))
    return items


def prepare_mt_bench_prompts(
    cfg: RunConfig, resolved_prompt: ResolvedJudgePrompt, _instructions: pd.DataFrame
) -> tuple[Callable, float | None]:
    """Resolve prompt variants before generation and prepare comparison batches."""
    spec = cfg.resolved_task.spec.protocol.judge
    settings = MTBenchPromptOptions.model_validate(spec.prompt_options)
    if cfg.judge.swap_mode not in ("fixed", "both"):
        raise ValueError(
            f"MT-Bench supports only swap_mode='fixed' or 'both'; got {cfg.judge.swap_mode!r}."
        )
    if (
        resolved_prompt.parser is not None
        and resolved_prompt.parser.requires_top_logprobs
    ):
        raise ValueError("MT-Bench does not support parsers that require logprobs.")
    if resolved_prompt.delegated:
        select = partial(
            _select_prompt,
            reference_categories=settings.reference_categories,
            prompt_preset=settings.fastchat_prompt_preset,
        )
        # Validate the selected compatibility preset before generation.
        select(None, False)
    else:
        prompts = build_mt_bench_preset_prompts(
            {
                multi: resolve_run_judge_prompt(
                    cfg.resolved_task, cfg.judge, multi_turn=multi
                )
                for multi, enabled in (
                    (False, spec.turns in ("all", "last") or 1 in spec.turns),
                    (True, spec.turns in ("all", "last") or 2 in spec.turns),
                )
                if enabled
            }
        )

        def select(category, multi):
            return prompts[multi, (category or "") in settings.reference_categories]

    rendering = dict(
        select_prompt=select,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
    )

    def prepare_batches(comparisons):
        items = build_mt_bench_items(comparisons, **rendering)
        return [
            (
                comparisons.index,
                [item.prompt for item in items],
                partial(render_mt_bench_requests, **rendering),
            )
        ]

    return prepare_batches, (
        settings.fastchat_temperature if resolved_prompt.delegated else None
    )
