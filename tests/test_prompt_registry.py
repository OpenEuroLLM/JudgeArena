from __future__ import annotations

from dataclasses import dataclass

import pytest

from judgearena.evaluate import resolve_judge_prompts, resolve_run_judge_prompt
from judgearena.prompts import resolve_judge_prompt as public_resolve_judge_prompt
from judgearena.prompts.registry import (
    DEFAULT_WITH_EXPLANATION_PRESET,
    FASTCHAT_PAIRWISE_PROMPT_PRESET,
    FLUENCY_JUDGE_PROMPT_PRESET,
    PRESETS,
    default_preset_for_task,
    resolve_judge_prompt,
)


@dataclass
class FakeCliArgs:
    prompt_preset: str | None = None
    prompt: object | None = None


def test_default_presets_are_owned_by_task_yaml():
    assert default_preset_for_task("alpaca-eval") == "default"
    assert default_preset_for_task("arena-hard-v2.0") == "default"


def test_default_preset_for_fluency_prefix():
    assert default_preset_for_task("fluency-french") == FLUENCY_JUDGE_PROMPT_PRESET
    assert default_preset_for_task("fluency-spanish") == FLUENCY_JUDGE_PROMPT_PRESET


def test_m_arena_hard_prompt_defaults_are_owned_by_task_yaml():
    assert default_preset_for_task("m-arena-hard-v0.1-uk") == "default"
    assert default_preset_for_task("m-arena-hard-v2.0-EU") == "default"


def test_default_preset_for_unknown_task():
    assert default_preset_for_task("new-benchmark") == "default"
    assert default_preset_for_task(None) == "default"


def test_mt_bench_default_is_delegated_fastchat():
    resolved = resolve_judge_prompt(task="mt-bench")

    assert resolved.preset_name == FASTCHAT_PAIRWISE_PROMPT_PRESET
    assert resolved.delegated is True
    assert resolved.system_prompt is None
    assert resolved.user_prompt_template == ""


def test_fluency_task_resolves_inline_system_prompt():
    resolved = resolve_judge_prompt(task="fluency-french")

    assert resolved.preset_name == FLUENCY_JUDGE_PROMPT_PRESET
    assert resolved.source == "preset"
    assert "completion of a sentence" in resolved.system_prompt
    assert "{user_prompt}" in resolved.user_prompt_template


def test_explicit_preset_wins_over_task_default():
    resolved = resolve_judge_prompt(
        task="mt-bench",
        preset="default",
    )

    assert resolved.preset_name == "default"
    assert resolved.delegated is False


def test_explanation_preset_renders_explanation_suffix():
    resolved = resolve_judge_prompt(preset=DEFAULT_WITH_EXPLANATION_PRESET)

    assert "explanation of your judgement" in resolved.user_prompt_template


def test_resolve_judge_prompts_honors_task_default_when_preset_omitted():
    resolved = resolve_judge_prompts(task="fluency-french")

    assert resolved.preset_name == FLUENCY_JUDGE_PROMPT_PRESET
    assert "completion of a sentence" in resolved.system_prompt


def test_prompts_package_reexports_registry_api():
    assert public_resolve_judge_prompt is resolve_judge_prompt


def test_unknown_preset_raises():
    with pytest.raises(KeyError, match="Unknown judge prompt preset"):
        resolve_judge_prompt(task="alpaca-eval", preset="does-not-exist")


def test_file_overrides_must_come_in_pair(tmp_path):
    system_file = tmp_path / "system.txt"
    system_file.write_text("Custom system", encoding="utf-8")

    with pytest.raises(ValueError, match="must be provided together"):
        resolve_judge_prompt(task="alpaca-eval", system_file=system_file)


def test_file_overrides_take_precedence_over_preset(tmp_path):
    system_file = tmp_path / "system.txt"
    user_file = tmp_path / "user.txt"
    system_file.write_text("Custom system", encoding="utf-8")
    user_file.write_text("Custom {completion_label}", encoding="utf-8")

    resolved = resolve_judge_prompt(
        task="alpaca-eval",
        preset=DEFAULT_WITH_EXPLANATION_PRESET,
        system_file=system_file,
        user_file=user_file,
    )

    assert resolved.source == "file"
    assert resolved.system_prompt == "Custom system"
    assert resolved.user_prompt_template == "Custom Answer"
    assert resolved.system_sha256 is not None
    assert resolved.user_sha256 is not None


def test_inline_prompt_overrides_keep_their_source():
    resolved = resolve_judge_prompts(
        system_prompt="Custom system",
        user_prompt_template="Custom {instruction}",
    )

    assert resolved.source == "override"


def test_file_overrides_accept_named_parser(tmp_path):
    from judgearena.prompts.parsing import JUDGE_PARSERS

    system_file = tmp_path / "system.txt"
    user_file = tmp_path / "user.txt"
    system_file.write_text("Judge with verdict labels", encoding="utf-8")
    user_file.write_text("Q: {user_prompt} A: {completion_A} B: {completion_B}")

    resolved = resolve_judge_prompt(
        system_file=system_file,
        user_file=user_file,
        parser="arena-hard-verdict",
    )

    assert resolved.parser is JUDGE_PARSERS["arena-hard-verdict"]
    assert resolved.parser("verdict: [[A>B]]") == 0.25
    assert resolved.metadata()["judge_parser"] == "arena-hard-verdict"


def test_named_parser_without_prompt_files_is_rejected():
    with pytest.raises(ValueError, match="requires judge prompt files"):
        resolve_judge_prompt(task="alpaca-eval", parser="score")


def test_unknown_named_parser_lists_available(tmp_path):
    system_file = tmp_path / "system.txt"
    user_file = tmp_path / "user.txt"
    system_file.write_text("s", encoding="utf-8")
    user_file.write_text("u", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown judge parser"):
        resolve_judge_prompt(
            system_file=system_file, user_file=user_file, parser="nope"
        )


def test_resolve_run_judge_prompt_reads_cli_fields():
    resolved_default = resolve_run_judge_prompt("alpaca-eval", FakeCliArgs())
    resolved_explain = resolve_run_judge_prompt(
        "alpaca-eval",
        FakeCliArgs(prompt_preset=DEFAULT_WITH_EXPLANATION_PRESET),
    )

    assert resolved_default.preset_name == "default"
    assert resolved_explain.preset_name == DEFAULT_WITH_EXPLANATION_PRESET


def test_every_preset_resolves_or_delegates():
    for preset_name, spec in PRESETS.items():
        resolved = resolve_judge_prompt(preset=preset_name)
        assert resolved.preset_name == preset_name
        if spec.delegated:
            assert resolved.delegated is True
        else:
            assert resolved.system_prompt
            assert resolved.user_prompt_template
