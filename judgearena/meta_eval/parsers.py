"""Winner and preference parsers for meta-evaluation prompt modes."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from judgearena.evaluate import PairScore

TIE_EPSILON = 0.01
META_EVAL_PAIRSCORE_TEMPERATURE = 0.5

_ARENA_HARD_PATTERNS = [
    re.compile(r"\[\[([AB<>=]+)\]\]"),
    re.compile(r"\[([AB<>=]+)\]"),
]
_ARENA_HARD_LIKERT_TO_WINNER = {
    "A>>B": "model_a",
    "A>B": "model_a",
    "A=B": "tie",
    "B>A": "model_b",
    "B>>A": "model_b",
    "B<<A": "model_a",
    "B<A": "model_a",
}


def pair_score_parser(
    temperature: float = META_EVAL_PAIRSCORE_TEMPERATURE,
) -> PairScore:
    parser = PairScore()
    parser.temperature = temperature
    return parser


def parse_pairscore_pref(judge_completion: str, *, temperature: float) -> float:
    score = pair_score_parser(temperature).parse_model_raw(judge_completion)
    if score is None or np.isnan(score):
        return 0.5
    return float(score)


def parse_pairscore_winner(
    judge_completion: str,
    *,
    temperature: float,
    eps: float = TIE_EPSILON,
) -> str:
    score = parse_pairscore_pref(judge_completion, temperature=temperature)
    if abs(score - 0.5) < eps:
        return "tie"
    if score > 0.5 + eps:
        return "model_b"
    if score < 0.5 - eps:
        return "model_a"
    return "tie"


def parse_arena_hard_winner(judge_completion: str) -> str:
    if not isinstance(judge_completion, str):
        return "tie"
    for pattern in _ARENA_HARD_PATTERNS:
        matches = pattern.findall(judge_completion.upper())
        matches = [match for match in matches if match]
        if matches:
            return _ARENA_HARD_LIKERT_TO_WINNER.get(matches[-1].strip(), "tie")
    return "tie"


def parse_alpaca_eval_winner(judge_completion: str) -> str:
    if not isinstance(judge_completion, str):
        return "tie"

    text = judge_completion
    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        obj_match = re.search(
            r'\{[^{}]*"ordered_models"[^{}]*\[[^\[\]]*\][^{}]*\}',
            text,
            re.DOTALL,
        )
        if obj_match:
            text = obj_match.group(0)

    try:
        data = json.loads(text)
        ordered = data.get("ordered_models", [])
        m_entry = next((entry for entry in ordered if entry.get("model") == "m"), None)
        if m_entry is None:
            return "tie"
        rank_m = m_entry["rank"]
        if rank_m == 1:
            return "model_a"
        if rank_m == 2:
            return "model_b"
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return "tie"


def winner_to_pref(winner: str) -> float:
    return {"model_a": 0.0, "model_b": 1.0}.get(winner, 0.5)


def parse_winner(judge_completion: str, prompt_mode: str) -> str:
    if prompt_mode == "arena-hard":
        return parse_arena_hard_winner(judge_completion)
    if prompt_mode == "alpaca-eval":
        return parse_alpaca_eval_winner(judge_completion)
    return parse_pairscore_winner(
        judge_completion,
        temperature=META_EVAL_PAIRSCORE_TEMPERATURE,
    )


def parse_pref(judge_completion: str, prompt_mode: str) -> float:
    if prompt_mode in ("arena-hard", "alpaca-eval"):
        return winner_to_pref(parse_winner(judge_completion, prompt_mode))
    return parse_pairscore_pref(
        judge_completion,
        temperature=META_EVAL_PAIRSCORE_TEMPERATURE,
    )


def add_parsed_columns(df: pd.DataFrame, prompt_mode: str) -> pd.DataFrame:
    out = df.copy()
    completions = out["judge_completion"].tolist()
    out["winner_llm"] = [
        parse_winner(completion, prompt_mode) for completion in completions
    ]
    out["pref_llm"] = [
        parse_pref(completion, prompt_mode) for completion in completions
    ]
    return out


def invert_winner(winner: str) -> str:
    if winner == "model_a":
        return "model_b"
    if winner == "model_b":
        return "model_a"
    return winner
