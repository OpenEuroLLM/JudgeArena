"""AlpacaEval token-logprob verdict parsing."""

from __future__ import annotations

from judgearena.prompts.parsing import (
    JudgeParser,
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


ALPACA_EVAL_TOKEN = AlpacaEvalToken()
