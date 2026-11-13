"""FastChat pairwise verdict parsing."""

from __future__ import annotations

from judgearena.prompts.parsing import JudgeParser, ParsedPreference
from judgearena.utils import strip_thinking_tags


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


VERDICT = FastChatVerdictParser()
