"""MT-Bench-101 upstream and cleaned score formats."""

from __future__ import annotations

import re

from judgearena.prompts.parsing import ParsedScore
from judgearena.prompts.registry import (
    MT_BENCH_101_CLEAN_PROMPT_PRESET,
    MT_BENCH_101_PROMPT_PRESET,
)

DOUBLE_BRACKET_PATTERN = re.compile(r"\[\[(\d+)\]\]")
UPSTREAM_SCORE_PATTERN = re.compile(r"\[([0-9]+)\]")


class MTBench101ScoreParser:
    """Parse the score format associated with an MT-Bench-101 prompt preset."""

    name = "mt-bench-101-score"
    result_type = ParsedScore
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


UPSTREAM = MTBench101ScoreParser()
CLEAN = MTBench101ScoreParser(MT_BENCH_101_CLEAN_PROMPT_PRESET)
