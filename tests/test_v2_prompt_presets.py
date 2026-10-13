"""Frozen v2 judge prompt presets: shared score contract, distinct guidance."""

from __future__ import annotations

from judgearena.prompts.registry import (
    SCORE_PARSER,
    V2_CRITERIA_JUDGE_PROMPT_PRESET,
    V2_EVIDENCE_JUDGE_PROMPT_PRESET,
    V2_VERIFICATION_JUDGE_PROMPT_PRESET,
    resolve_judge_prompt,
)

_V2 = (
    (V2_CRITERIA_JUDGE_PROMPT_PRESET, "instruction-relevant criteria"),
    (V2_EVIDENCE_JUDGE_PROMPT_PRESET, "evidence-based justification"),
    (V2_VERIFICATION_JUDGE_PROMPT_PRESET, "instruction satisfaction"),
)


def test_v2_presets_share_score_contract_and_differ():
    default_sys = resolve_judge_prompt(preset="default").system_prompt
    bodies = []
    for name, marker in _V2:
        r = resolve_judge_prompt(preset=name)
        assert r.parser is SCORE_PARSER and r.system_prompt == default_sys
        u = r.user_prompt_template
        assert "score_A:" in u and "score_B:" in u and "0 and 10" in u
        assert "{user_prompt}" in u and "{completion_A}" in u and "{completion_B}" in u
        assert marker in u
        bodies.append(u)
    assert len(set(bodies)) == 3
