"""Judge prompt registry resolution tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from judgearena.prompts.registry import (
    PRESETS,
    TASK_DEFAULT_PRESET,
    default_preset_for_task,
    resolve_judge_prompt,
)


def test_default_preset_for_task_known_keys():
    for task, preset in TASK_DEFAULT_PRESET.items():
        assert default_preset_for_task(task) == preset


def test_default_preset_for_fluency_prefix():
    """Any task starting with ``fluency-`` selects the fluency preset by default."""
    assert default_preset_for_task("fluency-french") == "fluency"
    assert default_preset_for_task("fluency-spanish") == "fluency"


def test_default_preset_for_m_arena_hard_prefix():
    """Per-language m-arena-hard tasks fall back to the default preset."""
    assert default_preset_for_task("m-arena-hard-EU") == "default"
    assert default_preset_for_task("m-arena-hard-fr") == "default"


def test_default_preset_for_unknown_task():
    """Anything not in TASK_DEFAULT_PRESET defaults to ``default``."""
    assert default_preset_for_task("brand-new-bench") == "default"
    assert default_preset_for_task(None) == "default"


def test_alpaca_eval_picks_default_preset():
    resolved = resolve_judge_prompt(task="alpaca-eval")
    assert resolved.name == "default"
    assert resolved.system_text  # bundled system prompt
    assert "{user_prompt}" in resolved.user_template_text


def test_mt_bench_picks_fastchat_pairwise_and_is_delegated():
    resolved = resolve_judge_prompt(task="mt-bench")
    assert resolved.name == "fastchat-pairwise"
    assert resolved.delegated is True
    # Delegated presets do not produce concrete prompt strings; the caller
    # uses its own prompt-selection machinery (FastChat-compatible).
    assert resolved.system_text == ""
    assert resolved.user_template_text == ""


def test_explicit_preset_wins_over_task_default():
    resolved = resolve_judge_prompt(
        task="alpaca-eval", preset="default_with_explanation"
    )
    assert resolved.name == "default_with_explanation"


def test_provide_explanation_legacy_alias_picks_explanation_preset():
    resolved = resolve_judge_prompt(task="alpaca-eval", provide_explanation=True)
    assert resolved.name == "default_with_explanation"


def test_unknown_preset_raises():
    with pytest.raises(KeyError, match="Unknown judge prompt preset"):
        resolve_judge_prompt(task="alpaca-eval", preset="does-not-exist")


def test_file_overrides_must_come_in_pair(tmp_path):
    sys_file = tmp_path / "sys.txt"
    sys_file.write_text("My system prompt", encoding="utf-8")
    with pytest.raises(ValueError, match="must be provided together"):
        resolve_judge_prompt(task="alpaca-eval", system_file=str(sys_file))


def test_file_overrides_take_precedence_over_preset(tmp_path):
    sys_file = tmp_path / "sys.txt"
    usr_file = tmp_path / "usr.txt"
    sys_file.write_text("My system prompt", encoding="utf-8")
    usr_file.write_text("My user prompt {completion_label}", encoding="utf-8")
    resolved = resolve_judge_prompt(
        task="alpaca-eval",
        preset="default_with_explanation",  # should be ignored
        system_file=str(sys_file),
        user_file=str(usr_file),
    )
    assert resolved.system_text == "My system prompt"
    assert "Answer" in resolved.user_template_text  # placeholder substituted
    assert resolved.name.startswith("file:")
    assert resolved.source == "file"


def test_resolve_run_judge_prompt_reads_from_cli_args():
    """``resolve_run_judge_prompt`` plucks the right knobs off a BaseCliArgs-shaped object."""
    from judgearena.evaluate import resolve_run_judge_prompt

    @dataclass
    class FakeArgs:
        judge_prompt_preset: str | None = None
        judge_system_prompt_file: str | None = None
        judge_user_prompt_file: str | None = None
        provide_explanation: bool = False

    resolved = resolve_run_judge_prompt("alpaca-eval", FakeArgs())
    assert resolved.name == "default"

    resolved_explain = resolve_run_judge_prompt(
        "alpaca-eval",
        FakeArgs(judge_prompt_preset="default_with_explanation"),
    )
    assert resolved_explain.name == "default_with_explanation"

    resolved_legacy = resolve_run_judge_prompt(
        "alpaca-eval", FakeArgs(provide_explanation=True)
    )
    assert resolved_legacy.name == "default_with_explanation"


def test_every_preset_round_trips_or_is_delegated():
    """Every entry in PRESETS resolves cleanly."""
    for name, spec in PRESETS.items():
        resolved = resolve_judge_prompt(preset=name)
        assert resolved.name == name
        if spec.delegated:
            assert resolved.delegated is True
        else:
            assert resolved.system_text  # non-empty
            assert resolved.user_template_text  # non-empty
