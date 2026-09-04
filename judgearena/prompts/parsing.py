"""Judge-output parsers; each prompt preset carries the parser for its format."""

from __future__ import annotations

import abc
import math
import re
from dataclasses import dataclass, field

from judgearena.utils import strip_thinking_tags


@dataclass(slots=True)
class ParsedPreference:
    """A canonical preference plus parser-specific evidence.

    ``preference`` is oriented to the judge input slots: 0 means A wins,
    0.5 means tie, and 1 means B wins. Parsers return ``None`` when the judge
    output cannot be parsed.
    """

    preference: float
    label: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.preference) or not 0 <= self.preference <= 1:
            raise ValueError("preference must be finite and between 0 and 1")


class JudgeParser(abc.ABC):
    """Parses judge output into a canonical preference and supporting evidence."""

    name: str
    """Registry key and run-metadata identifier."""

    requires_top_logprobs: bool = False
    """Whether judging should collect first-token top logprobs for this
    parser (the backend must also be asked for them via judge.top_logprobs)."""

    @abc.abstractmethod
    def __call__(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> float | None: ...

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None:
        """Wrap a scalar parser result for the structured parser interface."""
        preference = self(
            judge_completion,
            top_logprobs=top_logprobs,
        )
        return (
            None
            if preference is None
            else ParsedPreference(preference=float(preference))
        )


class PairScore(JudgeParser):
    """Score-format parser: temperature-softened preference from A/B scores."""

    name = "score"

    def __init__(self, *, temperature: float = 0.3):
        self.temperature = temperature

    def preference_from_scores(self, score_a: float, score_b: float) -> float:
        """Return a bounded preference without overflowing on extreme scores."""
        logit = self.temperature * (score_b - score_a)
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)

    def __call__(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> float | None:
        result = self.parse_result(
            judge_completion,
            top_logprobs=top_logprobs,
        )
        return None if result is None else result.preference

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None:
        score_a, score_b = self.parse_raw_scores(judge_completion)
        if score_a is None or score_b is None:
            return None
        return ParsedPreference(
            preference=float(self.preference_from_scores(score_a, score_b)),
            scores={"A": score_a, "B": score_b},
        )

    def parse_model_raw(self, judge_completion: str) -> float | None:
        """Return only the canonical preference for existing callers."""
        return self(judge_completion)

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


def parser_name(parse) -> str:
    """Short identifier of a parser for run metadata.

    Falls back to ``__name__`` for plain callables outside the registry
    (e.g. mt_bench's delegated FastChat parsers).
    """
    return getattr(parse, "name", getattr(parse, "__name__", "unknown"))


# Parsers selectable by name for runtime prompt overrides (judge.parser);
# presets reference these same instances.
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
