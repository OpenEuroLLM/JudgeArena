"""WildBench reward labels and absolute scores."""

from __future__ import annotations

from judgearena.prompts.parsing import (
    JudgeParser,
    ParsedPreference,
    ScoreParser,
    _json_judge_field,
)


class WildBenchVerdict(JudgeParser):
    """Parse WildBench's JSON choice, including its quoted-field fallback."""

    name = "wildbench-reward"
    _PREFERENCES = {"A++": 0.0, "A+": 0.25, "A=B": 0.5, "B+": 0.75, "B++": 1.0}

    def parse_result(
        self, judge_completion: str, *, top_logprobs: dict[str, float] | None = None
    ) -> ParsedPreference | None:
        details = _json_judge_field(judge_completion, "choice")
        label = details.get("choice")
        if not isinstance(label, str) or label not in self._PREFERENCES:
            return None
        return ParsedPreference(self._PREFERENCES[label], label=label, details=details)


REWARD = WildBenchVerdict()
SCORE = ScoreParser("wildbench-score", (1, 10))
