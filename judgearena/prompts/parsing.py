"""Judge-output parsers; each prompt preset carries the parser for its format."""

from __future__ import annotations

import abc
import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from judgearena.prompts.criteria import SCALE_MAX, SCALE_MIN
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


@dataclass(slots=True)
class ParsedScore:
    """A scalar judge score plus parser-specific evidence."""

    score: float
    label: str | None = None
    details: dict[str, object] = field(default_factory=dict)


class OutputParser[T](Protocol):
    """Parse a judge response without prescribing the result's meaning."""

    name: str
    result_type: type[T]
    """Runtime type of a successful parse; unparseable responses return None."""
    requires_top_logprobs: bool

    def parse_result(
        self, judge_completion: str, *, top_logprobs: dict[str, float] | None = None
    ) -> T | None: ...


def validate_parser(parser: OutputParser, expected: type | None = None) -> None:
    """Check the declared output before inference, without calling the parser."""
    result_type = getattr(parser, "result_type", None)
    if not isinstance(result_type, type):
        raise ValueError(f"Parser {parser_name(parser)!r} must declare a result_type.")
    if expected is not None and not issubclass(result_type, expected):
        raise ValueError(
            f"Parser {parser_name(parser)!r} must return {expected.__name__}; "
            f"declares {result_type.__name__}."
        )


class JudgeParser(abc.ABC):
    """Parses judge output into a canonical preference and supporting evidence."""

    name: str
    """Registry key and run-metadata identifier."""

    result_type = ParsedPreference

    requires_top_logprobs: bool = False
    """Whether this parser requires first-token top logprobs."""

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

    @abc.abstractmethod
    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None: ...


class FastChatVerdictParser(JudgeParser):
    name = "fastchat-verdict"

    def parse_result(
        self, judge_completion: str, *, top_logprobs: dict[str, float] | None = None
    ) -> ParsedPreference | None:
        text = strip_thinking_tags(judge_completion).strip()
        for token, preference, label in (
            ("A", 0.0, "A"),
            ("B", 1.0, "B"),
            ("C", 0.5, "tie"),
        ):
            if f"[[{token}]]" in text:
                return ParsedPreference(preference, label=label)
        return None


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

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None:
        text = strip_thinking_tags(judge_completion).upper()
        matches = [m for m in _ARENA_HARD_VERDICT_PATTERN.findall(text) if m]
        if not matches:
            matches = [
                m for m in _ARENA_HARD_VERDICT_FALLBACK_PATTERN.findall(text) if m
            ]
        if not matches:
            return None
        label = matches[-1].strip()
        preference = ARENA_HARD_VERDICT_PREFERENCES.get(label)
        if preference is None:
            return None
        return ParsedPreference(preference=preference, label=label)


def _json_judge_field(judge_completion: str, field: str) -> dict:
    """Read the last verdict object, retaining nested evidence after analysis/code."""
    text = strip_thinking_tags(judge_completion).strip()
    decoder = json.JSONDecoder()
    details = None
    offset = 0
    while (start := text.find("{", offset)) >= 0:
        try:
            value, offset = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, dict) and field in value:
            details = value
    if details is not None:
        return details
    # Some benchmark prompts permit a quoted verdict field without outer braces.
    matches = list(re.finditer(r'"' + re.escape(field) + r'"\s*:\s*', text))
    for match in reversed(matches):
        try:
            value, end = decoder.raw_decode(text, match.end())
            if end < len(text) and not text[end].isspace() and text[end] not in ",}]":
                continue
            return {field: value}
        except json.JSONDecodeError:
            continue
    return {}


def _numeric_score(
    value, score_range: tuple[float, float] | None = None
) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(score) or (
        score_range is not None and not score_range[0] <= score <= score_range[1]
    ):
        return None
    return score


class AlpacaEvalToken(JudgeParser):
    """Parse the official logprob-weighted AlpacaEval verdict.

    The annotator prompt labels the first answer "m" and the second answer "M".
    """

    name = "alpaca-eval-token"
    requires_top_logprobs = True
    _TOKENS = ("m", "M")

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None:
        if not top_logprobs:
            return None
        preference = weighted_token_preference(top_logprobs, self._TOKENS)
        if preference is None:
            return None
        label = strip_thinking_tags(judge_completion).strip()
        return ParsedPreference(
            preference=preference,
            label=label if label in self._TOKENS else None,
            scores={
                token: top_logprobs[token]
                for token in self._TOKENS
                if token in top_logprobs
            },
        )


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


def score_preference(
    score_a: float | None,
    score_b: float | None,
    *,
    temperature: float = 0.3,
) -> float | None:
    """0 favors A; 1 favors B. Missing scores give no preference."""
    if pd.isna(score_a) or pd.isna(score_b):
        return None
    logit = temperature * (score_b - score_a)
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


class PairScore(JudgeParser):
    """Score-format parser: temperature-softened preference from A/B scores."""

    name = "score"

    def __init__(self, *, temperature: float = 0.3):
        self.temperature = temperature

    def preference_from_scores(
        self,
        score_a: float | None,
        score_b: float | None,
        *,
        temperature: float | None = None,
    ) -> float | None:
        """Convert scores using a per-call or default temperature."""
        return score_preference(
            score_a,
            score_b,
            temperature=self.temperature if temperature is None else temperature,
        )

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
        temperature: float | None = None,
    ) -> ParsedPreference | None:
        score_a, score_b = self.parse_raw_scores(judge_completion)
        preference = self.preference_from_scores(
            score_a, score_b, temperature=temperature
        )
        if preference is None:
            return None
        return ParsedPreference(
            preference=preference,
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


def _criterion_names(names: list[str]) -> tuple[str, ...]:
    if (
        not names
        or len(set(names)) != len(names)
        or any(
            not name.strip() or name != name.strip() or "\n" in name or "\r" in name
            for name in names
        )
    ):
        raise ValueError("Expected nonempty, unique, single-line criterion names.")
    return tuple(names)


class CriteriaParser(PairScore):
    """Compare equally weighted criterion means, retaining every judge-slot score."""

    name = "criteria"

    def __init__(self, names: list[str], *, temperature: float = 0.3):
        super().__init__(temperature=temperature)
        self.names = _criterion_names(names)

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
        temperature: float | None = None,
    ) -> ParsedPreference | None:
        text = strip_thinking_tags(judge_completion).strip()
        names = {name.casefold(): name for name in self.names}
        criteria = {}
        for line in text.splitlines():
            line = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line)
            match = re.match(
                r"(.+?):\s*A\s*=\s*(\d+(?:\.\d+)?)\s*,"
                r"\s*B\s*=\s*(\d+(?:\.\d+)?)(?=\s|$)",
                line.replace("**", "").strip(),
                re.IGNORECASE,
            )
            if match is None:
                continue  # Ignore prose and code fences around the score lines.
            raw_name, raw_a, raw_b = match.groups()
            name = names.get(raw_name.strip().casefold())
            if name is None or name in criteria:
                return None
            scores = {"A": float(raw_a), "B": float(raw_b)}
            if any(not SCALE_MIN <= score <= SCALE_MAX for score in scores.values()):
                return None
            criteria[name] = scores
        if set(criteria) != set(self.names):
            return None
        means = {
            slot: sum(scores[slot] for scores in criteria.values()) / len(criteria)
            for slot in ("A", "B")
        }
        return ParsedPreference(
            preference=self.preference_from_scores(
                means["A"], means["B"], temperature=temperature
            ),
            scores=means,
            details={"criteria": criteria},
        )

    def parse_raw_scores(
        self, judge_completion: str
    ) -> tuple[float | None, float | None]:
        parsed = self.parse_result(judge_completion)
        return (
            (None, None) if parsed is None else (parsed.scores["A"], parsed.scores["B"])
        )


class MetaEvalPairScore(PairScore):
    """Parse complete integer score pairs in the meta-evaluation format."""

    name = "meta-eval-score"

    def __init__(self) -> None:
        super().__init__(temperature=0.5)

    @staticmethod
    def parse_raw_scores(
        judge_completion: str,
    ) -> tuple[float | None, float | None]:
        text = strip_thinking_tags(judge_completion).lower()

        def parse_score(label: str) -> float | None:
            match = re.search(
                rf'(?m)^[ \t\r]*["\']?score_{label}["\']?'
                rf"[ \t\r]*:[ \t\r]*([0-9]+)[ \t\r]*,?[ \t\r]*$",
                text,
            )
            if match is None:
                return None
            digits = match.group(1)
            if len(digits) > 2:
                return None
            score = int(digits)
            return float(score) if 0 <= score <= 10 else None

        return parse_score("a"), parse_score("b")


class AlpacaEvalJSON(JudgeParser):
    """Parse the ordered-model JSON emitted by the meta-eval Alpaca prompt."""

    name = "alpaca-eval-json"

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None:
        text = strip_thinking_tags(judge_completion)
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
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        ordered_models = data.get("ordered_models")
        if not isinstance(ordered_models, list) or len(ordered_models) != 2:
            return None

        ranks: dict[str, int] = {}
        for entry in ordered_models:
            if not isinstance(entry, dict):
                return None
            model = entry.get("model")
            rank = entry.get("rank")
            if not isinstance(model, str) or model not in {"m", "M"} or model in ranks:
                return None
            if type(rank) is not int or rank not in {1, 2}:
                return None
            ranks[model] = rank
        if ranks["m"] == ranks["M"]:
            return None

        winner = "m" if ranks["m"] == 1 else "M"
        return ParsedPreference(
            preference=0.0 if winner == "m" else 1.0,
            label=winner,
            details={"ranks": ranks},
        )


def parser_name(parse) -> str:
    """Short identifier of a parser for run metadata.

    Falls back to ``__name__`` for plain callables outside the registry
    (e.g. mt_bench's delegated FastChat parsers).
    """
    return getattr(parse, "name", getattr(parse, "__name__", "unknown"))


# Parsers selectable by name for runtime prompt overrides (judge.parser);
# presets reference these same instances.
JUDGE_PARSERS: dict[str, OutputParser] = {
    "score": PairScore(),
    "fastchat-verdict": FastChatVerdictParser(),
    "meta-eval-score": MetaEvalPairScore(),
    "arena-hard-verdict": ArenaHardVerdict(),
    "alpaca-eval-json": AlpacaEvalJSON(),
    "alpaca-eval-token": AlpacaEvalToken(),
}


def register_parser[T](name: str, parser: OutputParser[T]) -> OutputParser[T]:
    """Register a parser instance for use by name in custom prompt configs."""
    if not name.strip() or name in JUDGE_PARSERS:
        raise ValueError(f"Parser {name!r} is empty or already registered.")
    validate_parser(parser)
    JUDGE_PARSERS[name] = parser
    return parser


def resolve_judge_parser(name: str) -> OutputParser:
    """Return the registered judge parser named in a run config."""
    try:
        return JUDGE_PARSERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown judge parser {name!r}; available: {sorted(JUDGE_PARSERS)}"
        ) from exc
