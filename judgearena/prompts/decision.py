"""Configure structured decision questions and optional choice preferences."""

import math
from dataclasses import dataclass, replace
from functools import partial

from judgearena.inference import validate_decision_input
from judgearena.prompts.parsing import DecisionAnswersParser, ParsedPreference


def render_decision(row, *, state, questions):
    return {
        "state": {name: row[field] for name, field in state.items()},
        "questions": questions,
    }


@dataclass
class DecisionPreferenceParser(DecisionAnswersParser):
    question: str
    values: dict[str, float]
    name = "decision-preference"
    result_type = ParsedPreference

    def parse_result(self, judge_completion, *, top_logprobs=None):
        answers = super().parse_result(judge_completion, top_logprobs=top_logprobs)
        if answers is None:
            return None
        choice = answers.get(self.question, {}).get("choice")
        if not isinstance(choice, str) or choice not in self.values:
            return None
        return ParsedPreference(self.values[choice], label=choice, details=answers)


def configure_decision_prompt(
    prompt, *, questions=None, judging_mode="single_answer", state=None, preference=None
):
    if judging_mode not in ("single_answer", "pairwise"):
        raise ValueError("Decision judging_mode must be single_answer or pairwise.")
    validate_decision_input({"state": {}, "questions": questions})
    if state is None:
        state = {
            "instruction": "instruction",
            **(
                {"completion_a": "completion_a", "completion_b": "completion_b"}
                if judging_mode == "pairwise"
                else {"completion": "completion"}
            ),
        }
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(key, str) and key and isinstance(value, str) and value
            for key, value in state.items()
        )
    ):
        raise ValueError("Decision state must map names to prepared field names.")
    parser = DecisionAnswersParser()
    if judging_mode == "pairwise":
        if not isinstance(preference, dict) or set(preference) != {
            "question",
            "values",
        }:
            raise ValueError(
                "Pairwise decisions require preference.question and preference.values."
            )
        question = preference["question"]
        values = preference["values"]
        if (
            not isinstance(question, str)
            or question not in questions
            or questions[question].get("type") != "choice"
        ):
            raise ValueError("Decision preference must select a Choice question.")
        criteria = questions[question].get("criteria")
        if (
            not isinstance(values, dict)
            or not values
            or not isinstance(criteria, dict)
            or set(values) != set(criteria)
        ):
            raise ValueError(
                "Preference values must cover exactly the Choice criteria."
            )
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in values.values()
        ):
            raise ValueError(
                "Preference values must be finite numbers between 0 and 1."
            )
        parser = DecisionPreferenceParser(question, values)
    elif preference is not None:
        raise ValueError("Preference mapping requires pairwise judging_mode.")
    return replace(
        prompt,
        judging_mode=judging_mode,
        parser=parser,
        required_fields=tuple(state.values()),
        render_input=partial(render_decision, state=state, questions=questions),
    )
