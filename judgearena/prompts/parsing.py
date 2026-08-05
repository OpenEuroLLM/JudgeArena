"""Judge-output parsers; each prompt preset carries the parser for its format."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from judgearena.utils import strip_thinking_tags

JudgeParser = Callable[[str], "float | None"]


class PairScore:
    """Score-format parser: temperature-softened preference from A/B scores."""

    parser_name = "score"

    def __init__(self, *, temperature: float = 0.3):
        self.temperature = temperature

    def preference_from_scores(self, score_a: float, score_b: float) -> float:
        return 1 - np.exp(self.temperature * score_a) / (
            np.exp(self.temperature * np.array([score_a, score_b])).sum()
        )

    def __call__(self, judge_completion: str) -> float | None:
        return self.parse_model_raw(judge_completion)

    def parse_model_raw(self, judge_completion: str) -> float | None:
        score_a, score_b = self.parse_raw_scores(judge_completion)
        if score_a is None or score_b is None:
            return None
        return float(self.preference_from_scores(score_a, score_b))

    @staticmethod
    def parse_raw_scores(
        judge_completion: str,
    ) -> tuple[float | None, float | None]:
        """Extract the raw A and B scores from a judge completion (no temperature)."""
        # Strip thinking-model <think> blocks, then lower-case to avoid confusion
        # (e.g. when "a" is used instead of "A").
        text = strip_thinking_tags(judge_completion).lower()
        score_a = PairScore.get_regexp_match(text, r'score.*?a[": *\n]*(-?\d+)')
        score_b = PairScore.get_regexp_match(text, r'score.*?b[": *\n]*(-?\d+)')
        return score_a, score_b

    @staticmethod
    def get_regexp_match(s: str, regex: str, group_index: int = 1):
        m = re.search(re.compile(regex), s)
        if m is None:
            return None
        else:
            return float(m.group(group_index).strip(" "))


def parser_name(parse: JudgeParser) -> str:
    """Short identifier of a parser for run metadata."""
    return getattr(parse, "parser_name", getattr(parse, "__name__", "unknown"))


# Parsers selectable by name for runtime prompt overrides (judge.parser);
# presets reference their parser callable directly.
JUDGE_PARSERS: dict[str, JudgeParser] = {
    "score": PairScore(),
}


def resolve_judge_parser(name: str) -> JudgeParser:
    """Return the registered judge parser named in a run config."""
    try:
        return JUDGE_PARSERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown judge parser {name!r}; available: {sorted(JUDGE_PARSERS)}"
        ) from exc
