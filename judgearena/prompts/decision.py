"""Configure decision questions and their scalar or preference outcomes."""

import math
from dataclasses import dataclass, replace
from functools import partial

from judgearena.inference import validate_decision_input
from judgearena.prompts.parsing import (
    DecisionAnswersParser,
    ParsedPreference,
    ParsedScore,
)


def render_decision(row, *, state, questions):
    return {
        "state": {name: row[field] for name, field in state.items()},
        "questions": questions,
    }


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass
class DecisionResultParser(DecisionAnswersParser):
    question: str
    kind: str
    result_type: type
    invert: bool = False
    values: dict | None = None
    aggregation: str = "selected"
    name = "decision-result"

    def parse_result(self, judge_completion, *, top_logprobs=None):
        answers = super().parse_result(judge_completion, top_logprobs=top_logprobs)
        if answers is None:
            return None
        answer = answers.get(self.question, {})
        label = None
        if self.kind == "choice":
            label = answer.get("choice")
            if self.aggregation == "selected":
                if not isinstance(label, str) or label not in self.values:
                    return None
                value = self.values[label]
            else:
                probabilities = answer.get("probabilities")
                if (
                    not isinstance(probabilities, dict)
                    or set(probabilities) != set(self.values)
                    or any(
                        not _finite_number(p) or not 0 <= p <= 1
                        for p in probabilities.values()
                    )
                    or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6)
                ):
                    return None
                value = sum(
                    probabilities[key] * weight for key, weight in self.values.items()
                )
                value = min(1.0, max(0.0, value))
        else:
            value = answer.get(self.kind)
            if not _finite_number(value) or (
                self.kind == "noul" and not 0 <= value <= 1
            ):
                return None
            if self.invert:
                value = 1 - value
        return self.result_type(float(value), label=label, details=answers)


def _result_parser(result, questions, judging_mode):
    if result is None:
        if judging_mode == "pairwise":
            raise ValueError("Pairwise decisions require result.preference.")
        return DecisionAnswersParser()
    output = "preference" if judging_mode == "pairwise" else "score"
    if not isinstance(result, dict) or set(result) != {output}:
        raise ValueError(f"{judging_mode} decisions require result.{output}.")
    options = result[output]
    if not isinstance(options, dict):
        raise ValueError(
            "Decision result must define a question and conversion options."
        )
    question = options.get("question")
    if not isinstance(question, str) or question not in questions:
        raise ValueError("Decision result must select an existing question.")
    kind = questions[question].get("type")
    allowed = {"noul", "choice"} if output == "preference" else {"noul", "score"}
    if kind not in allowed:
        raise ValueError(f"Cannot convert {kind!r} to {output}.")
    keys = {"question"}
    if output == "preference":
        keys |= {"values", "aggregation"} if kind == "choice" else {"invert"}
    if set(options) - keys:
        raise ValueError(
            f"Unsupported decision result options: {sorted(set(options) - keys)}"
        )
    invert = options.get("invert", False)
    if not isinstance(invert, bool):
        raise ValueError("Decision invert must be boolean.")
    values = options.get("values")
    aggregation = options.get("aggregation", "selected")
    if kind == "choice":
        criteria = questions[question].get("criteria")
        if (
            not isinstance(values, dict)
            or not values
            or not isinstance(criteria, dict)
            or set(values) != set(criteria)
            or any(not _finite_number(v) or not 0 <= v <= 1 for v in values.values())
        ):
            raise ValueError(
                "Preference values must cover Choice criteria with finite numbers between 0 and 1."
            )
        if aggregation not in ("selected", "expected"):
            raise ValueError("Decision aggregation must be selected or expected.")
    return DecisionResultParser(
        question,
        kind,
        ParsedPreference if output == "preference" else ParsedScore,
        invert=invert,
        values=values,
        aggregation=aggregation,
    )


def configure_decision_prompt(
    prompt, *, questions=None, judging_mode="single_answer", state=None, result=None
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
    parser = _result_parser(result, questions, judging_mode)
    return replace(
        prompt,
        judging_mode=judging_mode,
        parser=parser,
        required_fields=tuple(state.values()),
        render_input=partial(render_decision, state=state, questions=questions),
    )
