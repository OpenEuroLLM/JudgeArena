"""Winner parsers for meta-eval prompt presets."""

from __future__ import annotations

import json
import re

import numpy as np

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


def parse_pairscore_pref(
    judge_completion: str, *, temperature: float = META_EVAL_PAIRSCORE_TEMPERATURE
) -> float:
    score = PairScore(temperature=temperature).parse_model_raw(judge_completion)
    if score is None or np.isnan(score):
        return 0.5
    return float(score)


def parse_pairscore_winner(
    judge_completion: str,
    *,
    temperature: float = META_EVAL_PAIRSCORE_TEMPERATURE,
    eps: float = TIE_EPSILON,
) -> str:
    score = parse_pairscore_pref(judge_completion, temperature=temperature)
    if abs(score - 0.5) <= eps:
        return "tie"
    return "model_b" if score > 0.5 else "model_a"


def parse_arena_hard_winner(judge_completion: str) -> str:
    if not isinstance(judge_completion, str):
        return "tie"
    for pattern in _ARENA_HARD_PATTERNS:
        matches = [
            match for match in pattern.findall(judge_completion.upper()) if match
        ]
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
        m_entry = next(
            (
                entry
                for entry in data.get("ordered_models", [])
                if entry.get("model") == "m"
            ),
            None,
        )
        if m_entry is None:
            return "tie"
        if m_entry["rank"] == 1:
            return "model_a"
        if m_entry["rank"] == 2:
            return "model_b"
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return "tie"


def winner_to_pref(winner: str) -> float:
    return {"model_a": 0.0, "model_b": 1.0}.get(winner, 0.5)


def parse_winner(judge_completion: str, prompt_preset: str) -> str:
    if prompt_preset == "arena-hard":
        return parse_arena_hard_winner(judge_completion)
    if prompt_preset == "alpaca-eval":
        return parse_alpaca_eval_winner(judge_completion)
    return parse_pairscore_winner(judge_completion)


def parse_pref(judge_completion: str, prompt_preset: str) -> float:
    if prompt_preset in ("arena-hard", "alpaca-eval"):
        return winner_to_pref(parse_winner(judge_completion, prompt_preset))
    return parse_pairscore_pref(judge_completion)
