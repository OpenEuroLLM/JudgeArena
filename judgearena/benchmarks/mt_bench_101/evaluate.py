"""Per-turn 1-10 judging and min-per-dialogue aggregation for MT-Bench-101."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.resources import files

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from judgearena.datasets.mt_bench_101 import (
    MT_BENCH_101_REFERENCE_TASKS,
    MT_BENCH_101_TASK_TO_ABILITY,
)
from judgearena.models import do_inference
from judgearena.prompts.parsing import PairScore, ParsedScore
from judgearena.prompts.registry import (
    MT_BENCH_101_CLEAN_PROMPT_PRESET,
    MT_BENCH_101_PROMPT_PRESET,
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

    def __init__(self, prompt_preset: str = MT_BENCH_101_PROMPT_PRESET) -> None:
        self.prompt_preset = prompt_preset

    def __call__(self, judge_completion: str) -> float | None:
        result = self.parse_result(judge_completion)
        return None if result is None else result.score

    def parse_result(self, judge_completion: str) -> ParsedScore | None:
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


def parse_mt_bench_101_rating(
    judge_completion: str,
    prompt_preset: str = MT_BENCH_101_PROMPT_PRESET,
) -> float | None:
    return MTBench101ScoreParser(prompt_preset)(judge_completion)


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
) -> tuple[str, str]:
    prompts = load_mt_bench_101_prompts(prompt_preset)
    if prompt_preset == MT_BENCH_101_PROMPT_PRESET:
        history = "".join(
            f"\n\n Human: {turn.get('user', '')}\n\nAssistant: {turn.get('bot', '')}"
            for turn in golden_context
        )
        history += f"\n\n Human: {user_message}\n\nAssistant: "
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
        user_prompt = prompts["user_prompt_templates"][template_name].format(
            history=history,
            prediction=assistant_message,
            reference_answer=upstream_reference,
        )
        return prompts["system_prompts"][task], user_prompt

    task_prompts = prompts["task_prompts"]
    dialogue = format_mt_bench_101_dialogue(
        golden_context=golden_context,
        user_message=user_message,
        assistant_message=assistant_message,
    )
    user_prompt = f"The dialogue need to be judged is: \n *** \n {dialogue} \n ***"
    if task in MT_BENCH_101_REFERENCE_TASKS:
        user_prompt += (
            f"\n\nThe reference solution is: \n ### \n {reference_answer} \n ###\n\n"
        )
    system_prompt = (
        f"{prompts['global_system']}\n\n"
        f"{task_prompts[task]}\n\n"
        f"{prompts['scoring_format']}"
    ).strip()
    return system_prompt, user_prompt


def judge_mt_bench_101_single(
    *,
    judge_chat_model,
    eval_items: pd.DataFrame,
    completions: pd.DataFrame,
    evaluated_model: str,
    prompt_preset: str = MT_BENCH_101_PROMPT_PRESET,
    truncate_input_chars: int | None = 8192,
    use_tqdm: bool = False,
    strip_thinking_before_judging: bool = False,
) -> pd.DataFrame:
    completion_by_idx = (
        completions
        if "instruction_index" not in completions.columns
        else completions.set_index("instruction_index")
    )
    rows: list[dict[str, object]] = []
    for idx in eval_items.index:
        eval_row = eval_items.loc[idx]
        completion_row = completion_by_idx.loc[idx]
        task = str(eval_row["task"])
        model_response = safe_text(completion_row.get("completion", ""), None)
        if strip_thinking_before_judging:
            model_response = strip_thinking_tags(model_response)
        model_response = safe_text(model_response, truncate_input_chars)
        system_prompt, user_prompt = build_mt_bench_101_judge_prompt(
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
                "instruction_index": idx,
                "dialogue_id": eval_row["dialogue_id"],
                "dialogue_uid": eval_row["dialogue_uid"],
                "task": task,
                "ability": eval_row.get("ability", MT_BENCH_101_TASK_TO_ABILITY[task]),
                "domain": eval_row["domain"],
                "turn_index": eval_row["turn_index"],
                "model_completion": model_response,
                "judge_prompt_preset": prompt_preset,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "{system_prompt}"), ("user", "{user_prompt}")]
    )
    inputs = prompt_template.batch(
        [
            {"system_prompt": row["system_prompt"], "user_prompt": row["user_prompt"]}
            for row in rows
        ]
    )
    judge_completions = do_inference(
        chat_model=judge_chat_model,
        inputs=inputs,
        use_tqdm=use_tqdm,
        stage="judging",
        cache_row_metadata=[
            {
                "instruction_id": f"{row['dialogue_uid']}:turn-{row['turn_index']}",
                "model_a": evaluated_model,
                "model_b": None,
            }
            for row in rows
        ],
    )
    for row, judge_completion in zip(rows, judge_completions, strict=True):
        row["judge_completion"] = judge_completion
        row["score"] = parse_mt_bench_101_rating(judge_completion, prompt_preset)
    return pd.DataFrame(rows)


def derive_mt_bench_101_pairwise_preferences(
    scored_a: pd.DataFrame,
    scored_b: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "instruction_index",
        "dialogue_uid",
        "dialogue_id",
        "task",
        "ability",
        "domain",
        "turn_index",
    ]
    merged = (
        scored_a.loc[:, cols + ["score"]]
        .rename(columns={"score": "score_A"})
        .merge(
            scored_b.loc[:, cols + ["score"]].rename(columns={"score": "score_B"}),
            on=cols,
            how="inner",
        )
    )
    scorer = PairScore()
    merged["preference"] = [
        None
        if pd.isna(score_a) or pd.isna(score_b)
        else float(scorer.preference_from_scores(score_a, score_b))
        for score_a, score_b in zip(merged["score_A"], merged["score_B"], strict=True)
    ]
    return merged


def aggregate_mt_bench_101_dialogues(pairwise_turns: pd.DataFrame) -> pd.DataFrame:
    """Apply the benchmark's minimum-turn rule before deriving preferences."""
    group_columns = ["dialogue_uid", "dialogue_id", "task", "ability", "domain"]
    dialogues = pairwise_turns.groupby(group_columns, as_index=False)[
        ["score_A", "score_B"]
    ].min()
    scorer = PairScore()
    dialogues["preference"] = [
        None
        if pd.isna(score_a) or pd.isna(score_b)
        else float(scorer.preference_from_scores(score_a, score_b))
        for score_a, score_b in zip(
            dialogues["score_A"], dialogues["score_B"], strict=True
        )
    ]
    return dialogues
