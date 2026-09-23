"""Jev-specific views of the packaged judge prompt presets."""

from judgearena.prompts.presets import PROMPT_PRESETS

JEV_PROMPT_PRESETS = {
    name: preset
    for name, preset in PROMPT_PRESETS.items()
    if preset.decision_mode is not None
}

JEV_HARD_TIE_THRESHOLDS = {
    preset.decision_mode: float(preset.hard_tie_threshold)
    for preset in JEV_PROMPT_PRESETS.values()
    if preset.hard_tie_threshold is not None
}
if any(not 0 <= threshold <= 1 for threshold in JEV_HARD_TIE_THRESHOLDS.values()):
    raise ValueError("Jev hard tie thresholds must be between 0 and 1.")

JEV_QUESTION_MODES: dict[str, dict[str, dict]] = {}
JEV_VERIFICATION_QUESTIONS: dict[str, dict[str, dict]] = {}
for preset in JEV_PROMPT_PRESETS.values():
    assert preset.decision_mode is not None and preset.questions is not None
    existing = JEV_QUESTION_MODES.setdefault(preset.decision_mode, preset.questions)
    if existing != preset.questions:
        raise ValueError(
            f"Jev presets using {preset.decision_mode!r} must define the same questions."
        )
    if preset.verification_questions is not None:
        JEV_VERIFICATION_QUESTIONS[preset.decision_mode] = preset.verification_questions
