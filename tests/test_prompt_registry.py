from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from judgearena.config import JudgeArgs
from judgearena.evaluate import (
    resolve_judge_prompts,
    resolve_run_judge_prompt,
)
from judgearena.prompts import resolve_judge_prompt as public_resolve_judge_prompt
from judgearena.prompts.registry import (
    DEFAULT_WITH_EXPLANATION_PRESET,
    FASTCHAT_PAIRWISE_PROMPT_PRESET,
    FLUENCY_JUDGE_PROMPT_PRESET,
    MT_BENCH_101_CLEAN_PROMPT_PRESET,
    MT_BENCH_101_PROMPT_PRESET,
    PRESETS,
    default_preset_for_task,
    resolve_judge_prompt,
)


def test_default_presets_are_owned_by_task_yaml():
    assert default_preset_for_task("alpaca-eval") == "alpaca-eval"
    assert default_preset_for_task("alpaca-eval-ja") == "default"
    assert default_preset_for_task("arena-hard-v2.0") == "arena-hard"


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


def test_mt_bench_101_prompt_presets_are_delegated():
    default = resolve_judge_prompt(task="mt-bench-101")
    cleaned = resolve_judge_prompt(preset=MT_BENCH_101_CLEAN_PROMPT_PRESET)

    assert default.preset_name == MT_BENCH_101_PROMPT_PRESET
    assert default.delegated is True
    assert cleaned.preset_name == MT_BENCH_101_CLEAN_PROMPT_PRESET
    assert cleaned.delegated is True


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

    assert resolved.preset_name == DEFAULT_WITH_EXPLANATION_PRESET
    assert "explanation of your judgement" in resolved.user_prompt_template
    assert "explanation of your judgement" in resolved.output_format
    assert resolved.user_prompt_template.endswith(resolved.output_format)


def test_resolve_judge_prompts_honors_task_default_when_preset_omitted():
    resolved = resolve_judge_prompts(task="fluency-french")

    assert resolved.preset_name == FLUENCY_JUDGE_PROMPT_PRESET
    assert "completion of a sentence" in resolved.system_prompt


def test_prompts_package_reexports_registry_api():
    assert public_resolve_judge_prompt is resolve_judge_prompt


def test_unknown_preset_raises():
    with pytest.raises(KeyError, match="Unknown judge prompt preset"):
        resolve_judge_prompt(task="alpaca-eval", preset="does-not-exist")


def test_system_file_requires_user_file(tmp_path):
    system_file = tmp_path / "system.txt"
    system_file.write_text("Custom system", encoding="utf-8")

    with pytest.raises(ValueError, match="user.*required"):
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


def test_file_overrides_accept_named_parser(tmp_path):
    from judgearena.prompts.parsing import JUDGE_PARSERS

    system_file = tmp_path / "system.txt"
    user_file = tmp_path / "user.txt"
    system_file.write_text("Judge with scores", encoding="utf-8")
    user_file.write_text("Q: {user_prompt} A: {completion_A} B: {completion_B}")

    resolved = resolve_judge_prompt(
        system_file=system_file,
        user_file=user_file,
        parser="score",
    )

    assert resolved.parser is JUDGE_PARSERS["score"]
    assert resolved.metadata()["judge_parser"] == "score"


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
    resolved_default = resolve_run_judge_prompt(
        "alpaca-eval", JudgeArgs(model="Dummy/j")
    )
    resolved_explain = resolve_run_judge_prompt(
        "alpaca-eval",
        JudgeArgs(model="Dummy/j", prompt={"preset": DEFAULT_WITH_EXPLANATION_PRESET}),
    )

    assert resolved_default.preset_name == "alpaca-eval"
    assert resolved_explain.preset_name == DEFAULT_WITH_EXPLANATION_PRESET


def test_every_preset_resolves_or_delegates():
    for preset_name, spec in PRESETS.items():
        resolved = resolve_judge_prompt(preset=preset_name)
        assert resolved.preset_name == preset_name
        if spec.delegated:
            assert resolved.delegated is True
        else:
            assert resolved.system_prompt is not None or spec.system_file is None
            assert resolved.user_prompt_template


@pytest.mark.parametrize("count", [None, 0, -1, True, 1.5, "5"])
def test_logprob_prompt_requires_a_positive_integer_count(count):
    settings = {"top_logprobs": count}
    resolve_judge_prompt(preset="default").validate_for(
        "pairwise", inference_kwargs=settings
    )
    with pytest.raises(ValueError, match="top_logprobs"):
        resolve_judge_prompt(preset="alpaca-eval").validate_for(
            "pairwise", inference_kwargs=settings
        )


def test_inline_prompt_registration_is_available_to_resolution(monkeypatch):
    from judgearena.prompts import (
        JudgePromptPreset,
        available_prompt_presets,
        register_prompt,
    )
    from judgearena.prompts.parsing import PairScore

    def configure(prompt, *, maximum=10):
        return replace(
            prompt,
            user_prompt_template=f"Grade {{completion}} from 1 to {maximum}.",
            parser=PairScore(temperature=maximum),
        )

    name = "test-inline-grammar"
    preset = JudgePromptPreset(
        name=name,
        inline_user="Grade {completion}.",
        configure=configure,
    )
    with monkeypatch.context() as patch:
        patch.setattr("judgearena.prompts.registry.PRESETS", PRESETS.copy())
        register_prompt(preset)
        resolved = resolve_judge_prompt(preset=name, options={"maximum": 5})
        assert name in available_prompt_presets()
        assert resolved.user_prompt_template == "Grade {completion} from 1 to 5."
        assert resolved.parser.parse_result(
            "Score_A: 8\nScore_B: 7"
        ).preference == pytest.approx(0.006692850924284856)
        assert resolved.parser.parse_result("no scores") is None
        assert resolved.user_sha256 != resolve_judge_prompt(preset=name).user_sha256
        from judgearena.config import RunConfig
        from judgearena.prompts.registry import group_judge_prompts
        from judgearena.tasks.registry import get_packaged_task

        task = get_packaged_task("alpaca-eval")
        spec = task.spec.model_dump()
        spec["protocol"]["judge"].update(
            default_prompt_preset=name, category_prompts={"creative": name}
        )
        task = replace(task, spec=type(task.spec).model_validate(spec))
        cfg = RunConfig(
            task=task.task,
            model={"name": "Dummy/a"},
            judge={
                "model": "Dummy/j",
                "prompt": {"preset": name, "options": {"maximum": 5}},
            },
        )
        cfg.resolved_task = task
        groups = group_judge_prompts(
            cfg,
            pd.DataFrame({"category": ["regular", "creative"]}),
            resolve_run_judge_prompt(task, cfg.judge),
        )
        assert all(
            prompt.parser.parse_result("Score_A: 8\nScore_B: 7").preference
            == pytest.approx(0.006692850924284856)
            for prompt, _ in groups
        )
        with pytest.raises(TypeError, match="unexpected keyword"):
            resolve_judge_prompt(preset=name, options={"maximumm": 5})
        with pytest.raises(ValueError, match="already registered"):
            register_prompt(preset)
