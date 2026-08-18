"""Parsers for the official WildBench judge formats."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

WildBenchParsedValue = float | str | None
WildBenchParser = Callable[[str], WildBenchParsedValue]


def _json_object(text: str) -> dict[str, object] | None:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1) if fenced is not None else stripped
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_wildbench_score(judge_completion: str) -> float | None:
    """Parse the official JSON score field and enforce its 1–10 range."""
    text = judge_completion.strip()
    value: object | None = None
    payload = _json_object(text)
    if payload is not None:
        value = payload.get("score")
    else:
        match = re.search(r'"score"\s*:\s*(?:"([^"]*?)"|(-?\d+(?:\.\d+)?))', text)
        if match is not None:
            value = match.group(1) or match.group(2)

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 10 else None


REWARD_FOR_A = {
    "A++": 1.0,
    "A+": 0.5,
    "A=B": 0.0,
    "B+": -0.5,
    "B++": -1.0,
}


def parse_wildbench_choice(judge_completion: str) -> str | None:
    """Parse one of the five official WB-Reward choices."""
    text = judge_completion.strip()
    value: object | None = None
    payload = _json_object(text)
    if payload is not None:
        value = payload.get("choice")
    else:
        match = re.search(r'"choice"\s*:\s*"([^"]*?)"', text)
        if match is not None:
            value = match.group(1)
    if not isinstance(value, str):
        return None
    choice = value.strip()
    return choice if choice in REWARD_FOR_A else None


WILDBENCH_PARSERS: dict[str, WildBenchParser] = {
    "wildbench-choice": parse_wildbench_choice,
    "wildbench-score": parse_wildbench_score,
}


def resolve_wildbench_parser(name: str) -> WildBenchParser:
    """Resolve the parser selected by a WildBench task definition."""
    try:
        return WILDBENCH_PARSERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown WildBench parser {name!r}; available: {sorted(WILDBENCH_PARSERS)}"
        ) from exc
