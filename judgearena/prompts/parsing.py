"""Judge-output parsers; each prompt preset carries the parser for its format."""

from __future__ import annotations

import abc
import math
import re

import numpy as np

from judgearena.utils import strip_thinking_tags


class JudgeParser(abc.ABC):
    """Parses what the judge returned into the universal preference.

    ``__call__`` receives the completion text plus, for parsers that declare
    ``requires_top_logprobs``, the first generated token's top logprobs — and
    returns the preference every pipeline consumes (0 = A wins, 0.5 = tie,
    1 = B wins, None = unparseable).
    """

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


# Graded preferences for the official Arena-Hard verdict labels. The spacing
# keeps decisiveness recoverable downstream (< 0.5 is an A win either way,
# but 0.0 marks a significant [[A>>B]] win vs 0.25 for [[A>B]]), and the
# encoding is symmetric so swapped-order judgments invert via 1 - preference.
ARENA_HARD_VERDICT_PREFERENCES: dict[str, float] = {
    "A>>B": 0.0,
    "A>B": 0.25,
    "B<<A": 0.0,
    "B<A": 0.25,
    "A=B": 0.5,
    "B=A": 0.5,
    "A<B": 0.75,
    "B>A": 0.75,
    "A<<B": 1.0,
    "B>>A": 1.0,
}

# Official Arena-Hard verdict extraction (judge_config.yaml regex_pattern),
# with v2.0's single-bracket fallback for judges that drop one bracket pair.
_ARENA_HARD_VERDICT_PATTERN = re.compile(r"\[\[([AB<>=]+)\]\]")
_ARENA_HARD_VERDICT_FALLBACK_PATTERN = re.compile(r"\[([AB<>=]+)\]")


class ArenaHardVerdict(JudgeParser):
    """Extract one graded verdict label, following the official rules.

    Like the current Arena-Hard-Auto ``get_score`` (the pipeline that governs
    v2.0): the judgment is uppercased and the LAST label found wins, so an
    explanation that mentions earlier labels still parses from its final
    verdict; only a judgment with no label at all is unparseable.
    """

    name = "arena-hard-verdict"

    def __call__(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> float | None:
        text = strip_thinking_tags(judge_completion).upper()
        matches = [m for m in _ARENA_HARD_VERDICT_PATTERN.findall(text) if m]
        if not matches:
            matches = [
                m for m in _ARENA_HARD_VERDICT_FALLBACK_PATTERN.findall(text) if m
            ]
        if not matches:
            return None
        return ARENA_HARD_VERDICT_PREFERENCES.get(matches[-1].strip())


class AlpacaEvalToken(JudgeParser):
    """Parse the official AlpacaEval verdict, weighted by logprobs when given.

    The annotator prompt labels the evaluated model "m" and the baseline "M".
    With top logprobs the preference is the official logprob weighting over
    the two tokens; without them it falls back to the sampled token (case is
    the whole signal, so matching is case-sensitive).
    """

    name = "alpaca-eval-token"
    requires_top_logprobs = True
    _TOKENS = ("m", "M")

    def __call__(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> float | None:
        if top_logprobs is not None:
            return weighted_token_preference(top_logprobs, self._TOKENS)
        token = strip_thinking_tags(judge_completion).strip()
        if token == "m":
            return 0.0
        if token == "M":
            return 1.0
        return None


def weighted_token_preference(
    top_logprobs: dict[str, float], tokens: tuple[str, str]
) -> float | None:
    """Official AlpacaEval logprob weighting over the two verdict tokens.

    Follows their ``logprob_parser``: a verdict token absent from the returned
    top logprobs counts as -inf (probability zero), and if both are absent the
    judgment is unparseable. Returns P(second token) renormalized over the
    pair, i.e. the preference for completion B.
    """
    logprob_a = top_logprobs.get(tokens[0])
    logprob_b = top_logprobs.get(tokens[1])
    if logprob_a is None and logprob_b is None:
        return None
    missing = float("-inf")
    scores = np.array(
        [
            logprob_a if logprob_a is not None else missing,
            logprob_b if logprob_b is not None else missing,
        ]
    )
    weights = np.exp(scores - scores.max())
    return float(weights[1] / weights.sum())


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
    "arena-hard-verdict": ArenaHardVerdict(),
    "alpaca-eval-token": AlpacaEvalToken(),
}


def resolve_judge_parser(name: str) -> JudgeParser:
    """Return the registered judge parser named in a run config."""
    try:
        return JUDGE_PARSERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown judge parser {name!r}; available: {sorted(JUDGE_PARSERS)}"
        ) from exc
