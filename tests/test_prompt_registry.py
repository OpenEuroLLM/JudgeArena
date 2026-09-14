from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from judgearena.evaluate import judge_and_parse_prefs, resolve_run_judge_prompt
from judgearena.models import DummyModel
from judgearena.prompts.parsing import JUDGE_PARSERS, PairScore
from judgearena.prompts.registry import (
    DEFAULT_WITH_EXPLANATION_PRESET,
    FASTCHAT_PAIRWISE_PROMPT_PRESET,
    FLUENCY_JUDGE_PROMPT_PRESET,
    PRESETS,
    default_preset_for_task,
    resolve_judge_prompt,
)


def test_default_preset_for_unknown_task():
    assert default_preset_for_task("new-benchmark") == "default"


def test_mt_bench_default_is_delegated_fastchat():
    resolved = resolve_judge_prompt(task="mt-bench")

    assert resolved.preset_name == FASTCHAT_PAIRWISE_PROMPT_PRESET
    assert resolved.delegated is True


def test_fluency_task_resolves_inline_system_prompt():
    resolved = resolve_judge_prompt(task="fluency-french")

    assert resolved.preset_name == FLUENCY_JUDGE_PROMPT_PRESET
    assert resolved.source == "preset"
    assert "completion of a sentence" in resolved.system_prompt
    assert "{user_prompt}" in resolved.user_prompt_template


def test_explicit_preset_wins_over_task_default():
    resolved = resolve_judge_prompt(task="mt-bench", preset="default")

    assert resolved.preset_name == "default"
    assert resolved.delegated is False


def test_explanation_preset_renders_explanation_suffix():
    resolved = resolve_judge_prompt(preset=DEFAULT_WITH_EXPLANATION_PRESET)

    assert "explanation of your judgement" in resolved.user_prompt_template


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
        parser="score",
    )

    assert resolved.parser is JUDGE_PARSERS["score"]
    assert resolved.metadata()["judge_parser"] == "score"
    assert resolved.source == "file"
    assert resolved.system_prompt == "Custom system"
    assert resolved.user_prompt_template == "Custom Answer"
    assert resolved.system_sha256 is not None
    assert resolved.user_sha256 is not None


def test_named_parser_without_prompt_files_is_rejected():
    with pytest.raises(ValueError, match="requires judge prompt files"):
        resolve_judge_prompt(task="alpaca-eval", parser="score")


def test_unknown_named_parser_lists_available(tmp_path):
    system_file = tmp_path / "system.txt"
    user_file = tmp_path / "user.txt"
    system_file.write_text("s", encoding="utf-8")
    user_file.write_text("u", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown judge parser.*score"):
        resolve_judge_prompt(
            system_file=system_file, user_file=user_file, parser="nope"
        )


def test_resolve_run_judge_prompt_reads_cli_fields():
    resolved_explain = resolve_run_judge_prompt(
        "alpaca-eval",
        SimpleNamespace(prompt_preset=DEFAULT_WITH_EXPLANATION_PRESET, prompt=None),
    )

    assert resolved_explain.preset_name == DEFAULT_WITH_EXPLANATION_PRESET


@pytest.mark.parametrize(
    ("parse", "expected"),
    [(None, 0.8807970779778823), (PairScore(temperature=0.5), 0.7310585786300049)],
)
def test_judging_uses_preset_parser_unless_overridden(monkeypatch, parse, expected):
    monkeypatch.setitem(
        PRESETS,
        "test-score",
        replace(PRESETS["default"], name="test-score", parser=PairScore(temperature=1)),
    )

    annotations, reversed_annotations, prefs = judge_and_parse_prefs(
        judge_chat_model=DummyModel("Dummy/score_A: 6\nscore_B: 8"),
        instructions=["Question"],
        completions_A=["Answer A"],
        completions_B=["Answer B"],
        prompt_preset="test-score",
        swap_mode="fixed",
        parse=parse,
    )

    assert annotations[0].parsed.preference == pytest.approx(expected)
    assert reversed_annotations is None
    assert prefs.tolist() == pytest.approx([expected])
