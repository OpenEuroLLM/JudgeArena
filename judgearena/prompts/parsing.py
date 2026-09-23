"""Judge-output parsers; each prompt preset carries the parser for its format."""

from __future__ import annotations

import abc
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from judgearena.prompts.jev import JEV_HARD_TIE_THRESHOLDS
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
        result = self.parse_result(judge_completion, top_logprobs=top_logprobs)
        return None if result is None else result.preference

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


class AlpacaEvalToken(JudgeParser):
    """Parse the official logprob-weighted AlpacaEval verdict.

    The annotator prompt labels the first answer "m" and the second answer "M".
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
        result = self.parse_result(judge_completion, top_logprobs=top_logprobs)
        return None if result is None else result.preference

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


class PairScore(JudgeParser):
    """Score-format parser: temperature-softened preference from A/B scores."""

    name = "score"

    def __init__(self, *, temperature: float = 0.3):
        self.temperature = temperature

    def preference_from_scores(
        self,
        score_a: float,
        score_b: float,
        *,
        temperature: float | None = None,
    ) -> float:
        """Return a bounded preference using a per-call or default temperature."""
        if temperature is None:
            temperature = self.temperature
        logit = temperature * (score_b - score_a)
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
        temperature: float | None = None,
    ) -> ParsedPreference | None:
        score_a, score_b = self.parse_raw_scores(judge_completion)
        if score_a is None or score_b is None:
            return None
        return ParsedPreference(
            preference=float(
                self.preference_from_scores(score_a, score_b, temperature=temperature)
            ),
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

    def __call__(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> float | None:
        result = self.parse_result(judge_completion, top_logprobs=top_logprobs)
        return None if result is None else result.preference

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


def _typesafe_probabilities(
    value: object, *, labels: set[str]
) -> dict[str, float] | None:
    try:
        probabilities = {
            label: float(probability) for label, probability in value.items()
        }
    except (AttributeError, TypeError, ValueError):
        return None
    if set(probabilities) != labels or any(
        not math.isfinite(probability) or probability < 0
        for probability in probabilities.values()
    ):
        return None
    total = sum(probabilities.values())
    # Jev rounds each displayed probability, so valid responses can total 0.99
    # or 1.01. Normalize that presentation error before scoring.
    if not math.isclose(total, 1.0, abs_tol=0.011):
        return None
    return {label: probability / total for label, probability in probabilities.items()}


class TypeSafeChoice(JudgeParser):
    """Parse Jev's A/B/tie probability distribution as a soft preference."""

    name = "typesafe-choice"

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None:
        try:
            result = json.loads(judge_completion)
            probabilities = _typesafe_probabilities(
                result["probabilities"], labels={"A", "B", "tie"}
            )
        except (KeyError, TypeError, ValueError):
            return None
        if probabilities is None:
            return None
        choice = result.get("choice")
        if choice not in probabilities:
            return None
        return ParsedPreference(
            preference=probabilities["B"] + 0.5 * probabilities["tie"],
            label=choice,
            scores=probabilities,
            details={
                key: result[key]
                for key in ("confidence", "model", "request_id")
                if result.get(key) is not None
            },
        )


class TypeSafeOverallChoice(JudgeParser):
    """Parse an overall A/B/tie/both-bad Choice from Jev."""

    name = "typesafe-overall-choice-v4"

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None:
        try:
            result = json.loads(judge_completion)
            answer = result["answers"]["outcome"]
            probabilities = _typesafe_probabilities(
                answer["probabilities"], labels={"A", "B", "tie", "both_bad"}
            )
            selection = answer["choice"]
        except (KeyError, TypeError, ValueError):
            return None
        if probabilities is None or selection not in probabilities:
            return None
        tie_probability = probabilities["tie"] + probabilities["both_bad"]
        decision_mode = result.get("decision_mode")
        hard_tie_threshold = JEV_HARD_TIE_THRESHOLDS.get(decision_mode)
        return ParsedPreference(
            preference=probabilities["B"] + 0.5 * tie_probability,
            label="tie" if selection == "both_bad" else selection,
            scores=probabilities,
            details={
                "outcome_selection": selection,
                "outcome_confidence": answer.get("confidence"),
                **(
                    {"hard_tie_threshold": hard_tie_threshold}
                    if hard_tie_threshold is not None
                    else {}
                ),
                **{
                    key: result[key]
                    for key in ("model", "request_id")
                    if result.get(key) is not None
                },
            },
        )


def _overall_score_preference(
    probabilities: dict[str, float],
) -> tuple[float, str]:
    preference = sum(level * probabilities[str(level)] for level in range(5)) / 4.0
    maximum = max(probabilities.values())
    winning_levels = {
        int(level)
        for level, probability in probabilities.items()
        if probability == maximum
    }
    spans_both_sides = any(level < 2 for level in winning_levels) and any(
        level > 2 for level in winning_levels
    )
    label = (
        "tie"
        if 2 in winning_levels or spans_both_sides
        else "A"
        if max(winning_levels) < 2
        else "B"
    )
    return preference, label


class TypeSafeOverallComparativeScore(JudgeParser):
    """Parse one overall five-level comparison while preserving its hard level."""

    name = "typesafe-overall-comparative-score-v5"
    level_labels = {str(level) for level in range(5)}

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None:
        try:
            result = json.loads(judge_completion)
            answer = result["answers"]["outcome"]
            probabilities = _typesafe_probabilities(
                answer["probabilities"], labels=self.level_labels
            )
        except (KeyError, TypeError, ValueError):
            return None
        if probabilities is None:
            return None
        preference, label = _overall_score_preference(probabilities)
        return ParsedPreference(
            preference=preference,
            label=label,
            scores=probabilities,
            details={
                "hard_preference_mode": "center_level",
                "outcome_score": answer.get("score"),
                "outcome_confidence": answer.get("confidence"),
                **{
                    key: result[key]
                    for key in ("model", "request_id")
                    if result.get(key) is not None
                },
            },
        )


class TypeSafeAbsoluteQualityScore(JudgeParser):
    """Compare two independent ten-level absolute quality distributions."""

    name = "typesafe-absolute-quality-score-v1"
    level_labels = {str(level) for level in range(10)}

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None:
        try:
            result = json.loads(judge_completion)
            answers = result["answers"]
            distributions = {
                candidate: _typesafe_probabilities(
                    answers[candidate]["probabilities"], labels=self.level_labels
                )
                for candidate in ("A", "B")
            }
        except (KeyError, TypeError, ValueError):
            return None
        if set(answers) != {"A", "B"} or any(
            distribution is None for distribution in distributions.values()
        ):
            return None
        probabilities_a = distributions["A"]
        probabilities_b = distributions["B"]
        assert probabilities_a is not None and probabilities_b is not None
        preference = sum(
            probabilities_a[str(level_a)]
            * probabilities_b[str(level_b)]
            * (1.0 if level_b > level_a else 0.5 if level_b == level_a else 0.0)
            for level_a in range(10)
            for level_b in range(10)
        )
        expected_scores = {
            candidate: 1
            + sum(level * distributions[candidate][str(level)] for level in range(10))
            for candidate in ("A", "B")
        }
        return ParsedPreference(
            preference=preference,
            label=(
                "tie"
                if math.isclose(expected_scores["A"], expected_scores["B"])
                else "A"
                if expected_scores["A"] > expected_scores["B"]
                else "B"
            ),
            scores=expected_scores,
            details={
                "hard_preference_mode": "absolute_quality",
                "score_scale": {"minimum": 1, "maximum": 10},
                "probabilities": distributions,
                "confidence": {
                    candidate: answers[candidate].get("confidence")
                    for candidate in ("A", "B")
                },
                **{
                    key: result[key]
                    for key in ("model", "request_id")
                    if result.get(key) is not None
                },
            },
        )


class TypeSafeVerdictSignals(JudgeParser):
    """Parse an overall verdict with reusable judgeability and Noul signals."""

    name = "typesafe-verdict-signals-v1"
    level_labels = {str(level) for level in range(5)}
    signal_ids = {
        f"{candidate}_{signal}"
        for candidate in ("A", "B")
        for signal in (
            "fulfills_core",
            "material_error",
            "useful_progress",
            "explicit_violation",
        )
    }
    route_labels = {
        "direct",
        "external_verification",
        "execution_required",
        "insufficient_context",
        "expert_review",
    }

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None:
        try:
            result = json.loads(judge_completion)
            answers = result["answers"]
            probabilities = _typesafe_probabilities(
                answers["outcome"]["probabilities"], labels=self.level_labels
            )
            route_probabilities = _typesafe_probabilities(
                answers["judgeability"]["probabilities"], labels=self.route_labels
            )
            route = answers["judgeability"]["choice"]
            signals = {
                answer_id: float(answers[answer_id]["noul"])
                for answer_id in self.signal_ids
            }
        except (KeyError, TypeError, ValueError):
            return None
        if (
            set(answers) != {"outcome", "judgeability", *self.signal_ids}
            or probabilities is None
            or route_probabilities is None
            or route not in self.route_labels
            or any(not 0 <= value <= 1 for value in signals.values())
        ):
            return None
        preference, label = _overall_score_preference(probabilities)
        return ParsedPreference(
            preference=preference,
            label=label,
            scores=probabilities,
            details={
                "hard_preference_mode": "center_level",
                "outcome_score": answers["outcome"].get("score"),
                "outcome_confidence": answers["outcome"].get("confidence"),
                "judgeability": route,
                "judgeability_probabilities": route_probabilities,
                "judgeability_confidence": answers["judgeability"].get("confidence"),
                "signals": signals,
                **{
                    key: result[key]
                    for key in ("model", "request_id")
                    if result.get(key) is not None
                },
            },
        )


class TypeSafeVerifiedVerdict(JudgeParser):
    """Parse a primary verdict conditionally accepted or revised by a second Jev call."""

    name = "typesafe-verified-verdict-v1"
    level_labels = {str(level) for level in range(5)}
    status_labels = {"accept", "revise", "escalate"}

    def parse_result(
        self,
        judge_completion: str,
        *,
        top_logprobs: dict[str, float] | None = None,
    ) -> ParsedPreference | None:
        try:
            result = json.loads(judge_completion)
            primary_answer = result["answers"]["outcome"]
            verification = result["verification"]
            status_answer = verification["status"]
            revised_answer = verification["revised_outcome"]
            primary = _typesafe_probabilities(
                primary_answer["probabilities"], labels=self.level_labels
            )
            revised = _typesafe_probabilities(
                revised_answer["probabilities"], labels=self.level_labels
            )
            status_probabilities = _typesafe_probabilities(
                status_answer["probabilities"], labels=self.status_labels
            )
            status = status_answer["choice"]
        except (KeyError, TypeError, ValueError):
            return None
        if (
            set(result["answers"]) != {"outcome"}
            or set(verification) != {"status", "revised_outcome"}
            or primary is None
            or revised is None
            or status_probabilities is None
            or status not in self.status_labels
            or status == "escalate"
        ):
            return None
        probabilities = primary if status == "accept" else revised
        preference, label = _overall_score_preference(probabilities)
        return ParsedPreference(
            preference=preference,
            label=label,
            scores=probabilities,
            details={
                "hard_preference_mode": "center_level",
                "verification_status": status,
                "verification_probabilities": status_probabilities,
                "verification_confidence": status_answer.get("confidence"),
                "primary_probabilities": primary,
                "revised_probabilities": revised,
                **{
                    key: result[key]
                    for key in ("model", "request_id")
                    if result.get(key) is not None
                },
            },
        )


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
    "meta-eval-score": MetaEvalPairScore(),
    "arena-hard-verdict": ArenaHardVerdict(),
    "alpaca-eval-json": AlpacaEvalJSON(),
    "alpaca-eval-token": AlpacaEvalToken(),
    "typesafe-choice": TypeSafeChoice(),
    "typesafe-overall-choice-v4": TypeSafeOverallChoice(),
    "typesafe-overall-comparative-score-v5": TypeSafeOverallComparativeScore(),
    "typesafe-absolute-quality-score-v1": TypeSafeAbsoluteQualityScore(),
    "typesafe-verdict-signals-v1": TypeSafeVerdictSignals(),
    "typesafe-verified-verdict-v1": TypeSafeVerifiedVerdict(),
}


def resolve_judge_parser(name: str) -> JudgeParser:
    """Return the registered judge parser named in a run config."""
    try:
        return JUDGE_PARSERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown judge parser {name!r}; available: {sorted(JUDGE_PARSERS)}"
        ) from exc
