"""AlpacaEval token, ranking and score-pair parsing."""

from __future__ import annotations

import json
import re

from judgearena.prompts.parsing import (
    JudgeParser,
    PairScore,
    ParsedPreference,
    weighted_token_preference,
)
from judgearena.utils import strip_thinking_tags


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


class AlpacaEvalJSON(JudgeParser):
    """Parse the ordered-model JSON emitted by the Alpaca ranking prompt."""

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


PAIR_SCORE = PairScore(temperature=0.5)
ALPACA_JSON = AlpacaEvalJSON()
ALPACA_EVAL_TOKEN = AlpacaEvalToken()
