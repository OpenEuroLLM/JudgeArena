"""Parsers for the meta-evaluation prompt variants."""

from __future__ import annotations

import json
import re

from judgearena.prompts.parsing import JudgeParser, PairScore, ParsedPreference
from judgearena.utils import strip_thinking_tags


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


PAIR_SCORE = MetaEvalPairScore()
ALPACA_JSON = AlpacaEvalJSON()
