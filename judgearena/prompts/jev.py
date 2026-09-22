"""Jev-specific views of the packaged judge prompt presets."""

from judgearena.prompts.presets import PROMPT_PRESETS, PromptPresetSpec

JEV_PROMPT_PRESETS = {
    name: preset
    for name, preset in PROMPT_PRESETS.items()
    if preset.decision_mode is not None
}

JEV_AGGREGATIONS = {
    preset.decision_mode: preset.aggregation
    for preset in JEV_PROMPT_PRESETS.values()
    if preset.aggregation is not None
}

_criteria_presets = [
    preset
    for preset in JEV_PROMPT_PRESETS.values()
    if preset.decision_mode == "criteria-score"
]
if len(_criteria_presets) != 1:
    raise ValueError("Exactly one Jev criteria-score preset must be defined.")
JEV_CRITERIA_SCORING = _criteria_presets[0].criteria_scoring
if not isinstance(JEV_CRITERIA_SCORING, dict):
    raise ValueError("The Jev criteria-score preset must define criteria_scoring.")


def _criteria_questions(scoring: dict) -> dict[str, dict]:
    candidates = scoring.get("candidates")
    criteria = scoring.get("criteria")
    question_template = scoring.get("question_template")
    boundary = scoring.get("boundary")
    if candidates != ["A", "B"] or not isinstance(criteria, list):
        raise ValueError(
            "Jev criteria scoring requires candidates A and B and criteria."
        )
    if not isinstance(question_template, str) or not isinstance(boundary, str):
        raise ValueError("Jev criteria scoring requires question and boundary text.")

    questions = {}
    for criterion in criteria:
        name = criterion.get("name")
        description = criterion.get("description")
        levels = criterion.get("levels")
        if (
            not isinstance(name, str)
            or not isinstance(description, str)
            or not isinstance(levels, list)
            or len(levels) < 2
            or any(
                not isinstance(level, dict)
                or not isinstance(level.get("value"), (int, float))
                or not isinstance(level.get("label"), str)
                for level in levels
            )
        ):
            raise ValueError(f"Invalid Jev criterion {name!r}.")
        values = [level["value"] for level in levels]
        if values != sorted(set(values)):
            raise ValueError(f"Jev criterion {name!r} levels must increase in value.")
        for candidate in candidates:
            questions[f"{candidate}_{name}"] = {
                "type": "score",
                "instructions": {
                    "question": question_template.format(
                        candidate=candidate, criterion_name=name
                    ),
                    "criterion": description,
                    "boundary": boundary,
                },
                "criteria": [level["label"] for level in levels],
            }
    return questions


def _questions_for_preset(preset: PromptPresetSpec) -> dict[str, dict]:
    assert preset.questions is not None
    questions = dict(preset.questions)
    if preset.decision_mode == "criteria-score":
        questions.update(_criteria_questions(JEV_CRITERIA_SCORING))
    return questions


JEV_QUESTION_MODES: dict[str, dict[str, dict]] = {}
for preset in JEV_PROMPT_PRESETS.values():
    assert preset.decision_mode is not None
    questions = _questions_for_preset(preset)
    existing = JEV_QUESTION_MODES.setdefault(preset.decision_mode, questions)
    if existing != questions:
        raise ValueError(
            f"Jev presets using {preset.decision_mode!r} must define the same questions."
        )
