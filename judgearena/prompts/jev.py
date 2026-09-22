"""Jev-specific views of the packaged judge prompt presets."""

from judgearena.prompts.presets import PROMPT_PRESETS

JEV_PROMPT_PRESETS = {
    name: preset
    for name, preset in PROMPT_PRESETS.items()
    if preset.decision_mode is not None
}

JEV_QUESTION_MODES: dict[str, dict[str, dict]] = {}
for preset in JEV_PROMPT_PRESETS.values():
    assert preset.decision_mode is not None and preset.questions is not None
    existing = JEV_QUESTION_MODES.setdefault(preset.decision_mode, preset.questions)
    if existing != preset.questions:
        raise ValueError(
            f"Jev presets using {preset.decision_mode!r} must define the same questions."
        )
