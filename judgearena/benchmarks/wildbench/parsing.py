"""Parsers for the official WildBench judge formats."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

WildBenchParser = Callable[[str], float | None]


def parse_wildbench_score(judge_completion: str) -> float | None:
    """Parse the official JSON score field and enforce its 1–10 range."""
    text = judge_completion.strip()
    value: object | None = None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            value = payload.get("score")
    except json.JSONDecodeError:
        match = re.search(r'"score"\s*:\s*"([^"]*?)"', text)
        if match is not None:
            value = match.group(1)

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 10 else None


WILDBENCH_PARSERS: dict[str, WildBenchParser] = {
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
