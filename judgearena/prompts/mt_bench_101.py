"""MT-Bench-101 task prompts, rendering, and scalar-score parsing."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from importlib.resources import files

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

from judgearena.datasets.mt_bench_101 import (
    MT_BENCH_101_REFERENCE_TASKS,
)
from judgearena.judging import JudgeRequest
from judgearena.prompts.parsing import ParsedScore
from judgearena.prompts.registry import (
    MT_BENCH_101_CLEAN_PROMPT_PRESET,
    MT_BENCH_101_PROMPT_PRESET,
    ResolvedJudgePrompt,
)
from judgearena.utils import safe_text, strip_thinking_tags

DOUBLE_BRACKET_PATTERN = re.compile(r"\[\[(\d+)\]\]")
UPSTREAM_SCORE_PATTERN = re.compile(r"\[([0-9]+)\]")

TASK_PROMPT_FILES = {
    "CM": "CM.txt",
    "AR": "AR.txt",
    "SI": "SI.txt",
    "TS": "TS.txt",
    "CC": "CC.txt",
    "CR": "rephrasing.txt",
    "FR": "rephrasing.txt",
    "SC": "SC.txt",
    "SA": "SA.txt",
    "MR": "MR.txt",
    "GR": "GR.txt",
    "IC": "IC.txt",
    "PI": "PI.txt",
}


def _prompt_text(name: str) -> str:
    return (
        files("judgearena.prompts")
        .joinpath("templates", "mt_bench_101", name)
        .read_text(encoding="utf-8")
    )


@lru_cache(maxsize=2)
def load_mt_bench_101_prompts(
    prompt_preset: str = MT_BENCH_101_PROMPT_PRESET,
) -> dict[str, object]:
    if prompt_preset == MT_BENCH_101_PROMPT_PRESET:
        return json.loads(_prompt_text("upstream.json"))
    if prompt_preset == MT_BENCH_101_CLEAN_PROMPT_PRESET:
        return {
            "global_system": _prompt_text("global_system.txt"),
            "scoring_format": _prompt_text("scoring_format.txt"),
            "task_prompts": {
                task: _prompt_text(prompt_file)
                for task, prompt_file in TASK_PROMPT_FILES.items()
            },
        }
    raise ValueError(f"Unsupported MT-Bench-101 prompt preset {prompt_preset!r}.")


class MTBench101ScoreParser:
    """Parse the score format associated with an MT-Bench-101 prompt preset."""

    name = "mt-bench-101-score"
    requires_top_logprobs = False

    def __init__(self, prompt_preset: str = MT_BENCH_101_PROMPT_PRESET) -> None:
        self.prompt_preset = prompt_preset

    def __call__(self, judge_completion: str) -> float | None:
        result = self.parse_result(judge_completion)
        return None if result is None else result.score

    def parse_result(
        self, judge_completion: str, *, top_logprobs: dict[str, float] | None = None
    ) -> ParsedScore | None:
        if self.prompt_preset == MT_BENCH_101_PROMPT_PRESET:
            match = UPSTREAM_SCORE_PATTERN.search(judge_completion)
            if match is None:
                return None
            return ParsedScore(score=float(match.group(1)), label=match.group(0))
        if self.prompt_preset != MT_BENCH_101_CLEAN_PROMPT_PRESET:
            raise ValueError(
                f"Unsupported MT-Bench-101 prompt preset {self.prompt_preset!r}."
            )
        for match in reversed(list(DOUBLE_BRACKET_PATTERN.finditer(judge_completion))):
            score = int(match.group(1))
            if 1 <= score <= 10:
                return ParsedScore(score=float(score), label=match.group(0))
        return None


def format_mt_bench_101_dialogue(
    *,
    golden_context: list[dict[str, str]],
    user_message: str,
    assistant_message: str,
) -> str:
    chunks: list[str] = []
    for turn in golden_context:
        chunks.append(
            f"\n\n Human: {turn.get('user', '')}\n\nAssistant: {turn.get('bot', '')}"
        )
    chunks.append(f"\n\n Human: {user_message}\n\nAssistant: {assistant_message}")
    return "".join(chunks)


def build_mt_bench_101_judge_prompt(
    *,
    task: str,
    golden_context: list[dict[str, str]],
    user_message: str,
    assistant_message: str,
    reference_answer: str,
    prompt_preset: str,
) -> tuple[ResolvedJudgePrompt, str]:
    """Return the selected template and its rendered user message."""
    prompts = load_mt_bench_101_prompts(prompt_preset)
    if prompt_preset == MT_BENCH_101_PROMPT_PRESET:
        history = format_mt_bench_101_dialogue(
            golden_context=golden_context,
            user_message=user_message,
            assistant_message="",
        )
        template_name = (
            "reference" if task in MT_BENCH_101_REFERENCE_TASKS else "default"
        )
        upstream_reference: str | list[dict[str, str]] = reference_answer
        if task in MT_BENCH_101_REFERENCE_TASKS:
            upstream_reference = [
                message
                for turn in golden_context
                for message in (
                    {"role": "user", "content": turn.get("user", "")},
                    {"role": "assistant", "content": turn.get("bot", "")},
                )
            ]
            upstream_reference.extend(
                (
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": reference_answer},
                )
            )
        system_prompt = prompts["system_prompts"][task]
        user_template = prompts["user_prompt_templates"][template_name]
        user_prompt = user_template.format(
            history=history,
            prediction=assistant_message,
            reference_answer=upstream_reference,
        )
    else:
        user_template = "The dialogue need to be judged is: \n *** \n {dialogue} \n ***"
        if task in MT_BENCH_101_REFERENCE_TASKS:
            user_template += (
                "\n\nThe reference solution is: \n ### \n {reference_answer} \n ###\n\n"
            )
        system_prompt = (
            f"{prompts['global_system']}\n\n"
            f"{prompts['task_prompts'][task]}\n\n"
            f"{prompts['scoring_format']}"
        ).strip()
        user_prompt = user_template.format(
            dialogue=format_mt_bench_101_dialogue(
                golden_context=golden_context,
                user_message=user_message,
                assistant_message=assistant_message,
            ),
            reference_answer=reference_answer,
        )
    return ResolvedJudgePrompt(
        preset_name=prompt_preset,
        variant_name=task,
        parser=MTBench101ScoreParser(prompt_preset),
        system_prompt=system_prompt,
        user_prompt_template=user_template,
        source="preset",
        system_sha256=hashlib.sha256(system_prompt.encode()).hexdigest(),
        user_sha256=hashlib.sha256(user_template.encode()).hexdigest(),
    ), user_prompt


def render_mt_bench_101_requests(
    samples: pd.DataFrame,
    *,
    prompt_preset: str = MT_BENCH_101_PROMPT_PRESET,
    truncate_input_chars: int | None = 8192,
    strip_thinking_before_judging: bool = False,
) -> tuple[pd.DataFrame, list[JudgeRequest[ParsedScore]], list[ResolvedJudgePrompt]]:
    """Prepare rendering fields, requests, and prompts from adapter turn rows.

    Samples contain completion text and the adapter's turn metadata. Returned
    annotation rows and request IDs follow the sample index.
    """
    rows: list[dict[str, object]] = []
    requests = []
    resolved_prompts = []
    for idx, eval_row in samples.iterrows():
        task = str(eval_row["task"])
        model_response = safe_text(eval_row["completion"], None)
        if strip_thinking_before_judging:
            model_response = strip_thinking_tags(model_response)
        model_response = safe_text(model_response, truncate_input_chars)
        prompt, user_prompt = build_mt_bench_101_judge_prompt(
            task=task,
            golden_context=list(eval_row.get("golden_context") or []),
            user_message=safe_text(
                eval_row.get("user_message", ""), truncate_input_chars
            ),
            assistant_message=model_response,
            reference_answer=safe_text(
                eval_row.get("reference_answer"), truncate_input_chars
            ),
            prompt_preset=prompt_preset,
        )
        rows.append(
            {
                "model_completion": model_response,
                "system_prompt": prompt.system_prompt,
                "user_prompt": user_prompt,
            }
        )
        requests.append(
            JudgeRequest(
                id=str(idx),
                input=ChatPromptValue(
                    messages=[
                        SystemMessage(content=prompt.system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                ),
                parser=prompt.parser,
            )
        )
        resolved_prompts.append(prompt)
    return pd.DataFrame(rows), requests, resolved_prompts
