"""Packaged official WildBench V2 task tests."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from judgearena.benchmarks.wildbench import runner
from judgearena.benchmarks.wildbench.prompting import (
    EMPTY_RESPONSE,
    render_reward_prompt,
    render_score_prompt,
)
from judgearena.config import RunConfig
from judgearena.tasks.registry import get_packaged_task

OFFICIAL_SCORE_PROMPT_SHA256 = (
    "c5984ae77b009f8dad0e8e37093348345f4e8d3099ce812bd7799ef7b73f005f"
)
OFFICIAL_REWARD_PROMPT_SHA256 = (
    "ccd57bd8c4c73f4f83cf8963ef3c2697c1c7b9e907ead91e0d0512cca4ae7a11"
)


def _examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instruction_index": ["session-1"],
            "instruction": ["Follow-up question"],
            "conversation_input": [
                [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Follow-up question"},
                ]
            ],
            "history": ["USER: First question\n\nASSISTANT: First answer\n\n"],
            "checklist": [["Is it correct?", "Is it clear?"]],
            "task_categories": [["Math & Data Analysis"]],
        }
    )


def test_official_score_task_pins_upstream_inputs_and_defaults():
    task = get_packaged_task("wildbench-v2-score-official")
    assert task is not None
    prompt = task.prompt_texts[task.spec.protocol.judge.prompt_file]

    assert task.spec.dataset.sources["examples"].revision == (
        "26c49eb39d7d5ce2099b0bbafed5a88dcce954ec"
    )
    assert hashlib.sha256(prompt.encode()).hexdigest() == (OFFICIAL_SCORE_PROMPT_SHA256)
    assert task.spec.protocol.judge.reference_judge == "gpt-4o-2024-05-13"
    assert task.spec.protocol.judge.max_words_to_eval is None


def test_wildbench_task_defaults_are_applied_to_run_config():
    cfg = RunConfig(
        task="wildbench-v2-score-official",
        model={"name": "Dummy/candidate"},
        judge={"model": "Dummy/judge"},
    )

    assert cfg.model.temperature == 0.0
    assert cfg.model.top_p == 1.0
    assert cfg.model.max_out_tokens == 4096
    assert cfg.generation.truncate_all_input_chars is None
    assert cfg.judge.temperature == 0.0
    assert cfg.judge.max_out_tokens == 1024


def test_official_reward_task_pins_references_prompt_and_defaults():
    task = get_packaged_task("wildbench-v2-reward-official")
    assert task is not None
    protocol = task.spec.protocol
    prompt = task.prompt_texts[protocol.judge.prompt_file]

    assert task.spec.dataset.sources["official_outputs"].revision == (
        "d6755bc68220df853c0825a733430f73f5af2501"
    )
    assert protocol.baseline.references == (
        "gpt-4-turbo-2024-04-09",
        "claude-3-haiku-20240307",
        "Llama-2-70b-chat-hf",
    )
    assert hashlib.sha256(prompt.encode()).hexdigest() == OFFICIAL_REWARD_PROMPT_SHA256
    assert protocol.judge.reference_judge == "gpt-4-turbo-2024-04-09"
    assert protocol.judge.max_words_to_eval == 1000
    assert protocol.judge.assignment_seed == 42

    cfg = RunConfig(
        task="wildbench-v2-reward-official",
        model={"name": "Dummy/candidate"},
        judge={"model": "Dummy/judge"},
    )
    assert cfg.model.baseline is None
    assert cfg.wildbench is not None
    assert cfg.wildbench.length_penalty_chars is None


def test_wildbench_score_prompt_contains_official_context_fields():
    task = get_packaged_task("wildbench-v2-score-official")
    prompt = task.prompt_texts[task.spec.protocol.judge.prompt_file]

    rendered = render_score_prompt(
        prompt,
        _examples().iloc[0],
        "Candidate answer",
        max_words=None,
        max_chars=None,
    )

    assert "USER: First question" in rendered
    assert "Follow-up question" in rendered
    assert "Candidate answer" in rendered
    assert "- Is it correct?\n- Is it clear?\n" in rendered
    assert not any(
        placeholder in rendered
        for placeholder in (
            "{$history}",
            "{$user_query}",
            "{$model_output}",
            "{$checklist}",
        )
    )


def test_wildbench_reward_prompt_preserves_orientation_and_visible_outputs():
    task = get_packaged_task("wildbench-v2-reward-official")
    prompt = task.prompt_texts[task.spec.protocol.judge.prompt_file]

    rendered = render_reward_prompt(
        prompt,
        _examples().iloc[0],
        "candidate",
        "",
        candidate_is_a=False,
        max_words=1000,
        max_chars=None,
    )

    assert rendered.candidate_output == "candidate"
    assert rendered.baseline_output == EMPTY_RESPONSE
    assert rendered.text.index(EMPTY_RESPONSE) < rendered.text.index("candidate")
    assert not any(
        placeholder in rendered.text
        for placeholder in (
            "{$history}",
            "{$user_query}",
            "{$candidate_A}",
            "{$candidate_B}",
            "{$checklist}",
        )
    )


def test_wildbench_reward_prompt_uses_official_word_truncation_marker():
    task = get_packaged_task("wildbench-v2-reward-official")
    prompt = task.prompt_texts[task.spec.protocol.judge.prompt_file]

    rendered = render_reward_prompt(
        prompt,
        _examples().iloc[0],
        "one two three",
        "reference",
        candidate_is_a=True,
        max_words=2,
        max_chars=None,
    )

    assert rendered.candidate_output == "one two... (truncated)"


def test_wildbench_score_run_writes_reproducible_artifacts(tmp_path, monkeypatch):
    examples = _examples()
    adapter = SimpleNamespace(load_instructions=lambda *_args: examples)
    monkeypatch.setattr(runner, "resolve_dataset_adapter", lambda _name: adapter)
    monkeypatch.setattr(
        runner,
        "_load_or_generate_outputs",
        lambda _cfg, frame: pd.Series(
            ["Candidate answer"],
            index=pd.Index(frame["instruction_index"].astype(str)),
        ),
    )
    monkeypatch.setattr(runner, "build_judge", lambda _cfg: object())
    monkeypatch.setattr(
        runner,
        "do_inference",
        lambda _model, inputs, **_kwargs: ['{"score": "8"}'] * len(inputs),
    )

    result = runner.run_wildbench(
        RunConfig(
            task="wildbench-v2-score-official",
            model={"name": "Dummy/candidate"},
            judge={"model": "Dummy/judge"},
            run={"result_folder": str(tmp_path), "no_log_file": True},
        )
    )

    result_path = Path(result["result_path"])
    saved = json.loads(result_path.read_text())
    assert result["wb_score"] == 6.0
    assert saved["wb_score_leaderboard"] == 60.0
    assert result["reference_judge"] == "gpt-4o-2024-05-13"
    assert (result_path.parent / "annotations.parquet").exists()
    assert (result_path.parent / "model_outputs.parquet").exists()
    assert (result_path.parent / "config.yaml").exists()
    assert (result_path.parent / "run-metadata.v1.json").exists()


def test_wildbench_score_rejects_runtime_baseline():
    with pytest.raises(ValueError, match="baseline is not used"):
        RunConfig(
            task="wildbench-v2-score-official",
            model={"name": "Dummy/candidate", "baseline": "Dummy/baseline"},
            judge={"model": "Dummy/judge"},
        )
