import hashlib
import json
import re

import pandas as pd
import pytest

import judgearena.benchmarks.mt_bench_101.runner as runner
import judgearena.datasets.mt_bench_101 as mt_bench_101
import judgearena.models as models
from judgearena.benchmarks.mt_bench_101.evaluate import (
    MTBench101ScoreParser,
    aggregate_mt_bench_101_dialogues,
    build_mt_bench_101_judge_prompt,
    derive_mt_bench_101_pairwise_preferences,
    judge_mt_bench_101_single,
    load_mt_bench_101_prompts,
    parse_mt_bench_101_rating,
)
from judgearena.benchmarks.mt_bench_101.generate import _build_golden_context_input
from judgearena.benchmarks.mt_bench_101.runner import run_mt_bench_101_benchmark
from judgearena.benchmarks.scoring import build_metric
from judgearena.cache.sqlite import (
    COMPLETION_DB_NAME,
    JUDGEMENT_DB_NAME,
    CompletionCache,
    JudgementCache,
)
from judgearena.config import RunConfig
from judgearena.datasets.mt_bench_101 import expand_mt_bench_101_records
from judgearena.models import DummyModel
from judgearena.prompts.registry import (
    MT_BENCH_101_CLEAN_PROMPT_PRESET,
    MT_BENCH_101_PROMPT_PRESET,
)
from judgearena.tasks.registry import get_packaged_task


def test_expand_mt_bench_101_turn_rules():
    records = [
        {
            "task": "CM",
            "id": 1,
            "history": [
                {"user": "u1", "bot": "b1"},
                {"user": "u2", "bot": "b2"},
            ],
        },
        {
            "task": "PI",
            "id": 2,
            "history": [
                {"user": "x1", "bot": "y1"},
                {"user": "x2", "bot": "y2"},
            ],
        },
        {
            "task": "MR",
            "id": 3,
            "history": [{"user": "q", "bot": "ref"}],
        },
    ]
    eval_items = expand_mt_bench_101_records(records)
    assert len(eval_items) == 4
    cm_rows = eval_items[eval_items["task"] == "CM"]
    assert cm_rows.iloc[0]["turn_index"] == 2
    assert len(cm_rows.iloc[0]["golden_context"]) == 1
    mr_rows = eval_items[eval_items["task"] == "MR"]
    assert mr_rows.iloc[0]["requires_reference"]
    assert mr_rows.iloc[0]["reference_answer"] == "ref"
    assert mr_rows.iloc[0]["ability"] == "reasoning"
    assert mr_rows.iloc[0]["domain"] == "adaptability"


def test_expand_mt_bench_101_unknown_task():
    with pytest.raises(ValueError, match="Unknown MT-Bench-101 task"):
        expand_mt_bench_101_records([{"task": "XX", "id": 1, "history": []}])


def test_mt_bench_101_generation_uses_only_upstream_dialogue_messages():
    prompt = _build_golden_context_input(
        system_prompt=None,
        golden_context=[{"user": "u1", "bot": "b1"}],
        user_message="u2",
        truncate_input_chars=None,
    )
    assert [(message.type, message.content) for message in prompt.to_messages()] == [
        ("human", "u1"),
        ("ai", "b1"),
        ("human", "u2"),
    ]


def test_download_mt_bench_101_uses_pinned_revision(tmp_path, monkeypatch):
    task = get_packaged_task("mt-bench-101")
    assert task is not None
    calls = {}

    def _urlretrieve(url, dest):
        calls["url"] = url
        dest.write_text("{}\n")

    monkeypatch.setattr(mt_bench_101, "urlretrieve", _urlretrieve)
    mt_bench_101.download_task_sources(task, tmp_path)
    assert "bc18b3e2c18c99164e11528f1a79c92083db5953" in calls["url"]
    assert calls["url"].endswith("data/subjective/mtbench101.jsonl")


def test_load_task_instructions_expands_cached_jsonl(tmp_path, monkeypatch):
    task = get_packaged_task("mt-bench-101")
    assert task is not None
    dataset_path = tmp_path / "_sources" / "mt-bench-101" / "mtbench101.jsonl"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        json.dumps(
            {
                "task": "PI",
                "id": 9,
                "history": [{"user": "hello", "bot": "world"}],
            }
        )
        + "\n"
    )
    monkeypatch.setattr(mt_bench_101, "download_task_sources", lambda *_args: None)
    loaded = mt_bench_101.load_task_instructions(task, tmp_path)
    assert loaded.iloc[0]["dialogue_uid"] == "PI:9"
    assert loaded.iloc[0]["instruction"] == "hello"


def test_parse_mt_bench_101_rating():
    assert parse_mt_bench_101_rating("Reasoning...\nRating: [[7]]") == pytest.approx(
        7.0
    )
    reasoning_output = "<think>Perhaps [[2]]...</think>\nRating: [[9]]"
    assert parse_mt_bench_101_rating(reasoning_output) == pytest.approx(2.0)
    assert MTBench101ScoreParser(MT_BENCH_101_CLEAN_PROMPT_PRESET).parse_result(
        reasoning_output
    ).score == pytest.approx(9.0)
    assert parse_mt_bench_101_rating("[[7]] then invalid [[11]]") == pytest.approx(7.0)
    assert parse_mt_bench_101_rating("Rating: [[0]]") == pytest.approx(0.0)
    assert parse_mt_bench_101_rating("Rating: [6]") == pytest.approx(6.0)
    assert (
        parse_mt_bench_101_rating("Rating: [6]", MT_BENCH_101_CLEAN_PROMPT_PRESET)
        is None
    )


def test_mt_bench_101_default_prompt_matches_pinned_upstream():
    prompts = load_mt_bench_101_prompts(MT_BENCH_101_PROMPT_PRESET)
    system_prompts = json.dumps(
        prompts["system_prompts"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    user_templates = json.dumps(
        prompts["user_prompt_templates"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert prompts["source_revision"] == "da7b1c3a007efea1dcca985a242e13a0ba51abd1"
    assert hashlib.sha256(system_prompts.encode()).hexdigest() == (
        "0314a328dfe4385873170a4047b5225a039b273b14ee80c10c8233f7e94c4a40"
    )
    assert hashlib.sha256(user_templates.encode()).hexdigest() == (
        "a440b1776edc1a2e4da4ed90101e66a2e7f47e143e3350a31f8ccbb8a0e7adc9"
    )

    system_prompt, user_prompt = build_mt_bench_101_judge_prompt(
        task="SA",
        golden_context=[{"user": "u1", "bot": "b1"}],
        user_message="u2",
        assistant_message="candidate",
        reference_answer="reference",
        prompt_preset=MT_BENCH_101_PROMPT_PRESET,
    )
    assert system_prompt == prompts["system_prompts"]["SA"]
    assert user_prompt == (
        "The dialogue need to be judged is: \n *** \n "
        "\n\n Human: u1\n\nAssistant: b1"
        "\n\n Human: u2\n\nAssistant:  candidate \n ***"
    )
    _, reference_user_prompt = build_mt_bench_101_judge_prompt(
        task="MR",
        golden_context=[{"user": "u1", "bot": "b1"}],
        user_message="u2",
        assistant_message="candidate",
        reference_answer="reference",
        prompt_preset=MT_BENCH_101_PROMPT_PRESET,
    )
    assert (
        "[{'role': 'user', 'content': 'u1'}, "
        "{'role': 'assistant', 'content': 'b1'}, "
        "{'role': 'user', 'content': 'u2'}, "
        "{'role': 'assistant', 'content': 'reference'}]"
    ) in reference_user_prompt


def test_mt_bench_101_clean_prompt_remains_selectable():
    upstream = build_mt_bench_101_judge_prompt(
        task="SA",
        golden_context=[],
        user_message="question",
        assistant_message="answer",
        reference_answer="",
        prompt_preset=MT_BENCH_101_PROMPT_PRESET,
    )
    cleaned = build_mt_bench_101_judge_prompt(
        task="SA",
        golden_context=[],
        user_message="question",
        assistant_message="answer",
        reference_answer="",
        prompt_preset=MT_BENCH_101_CLEAN_PROMPT_PRESET,
    )
    assert cleaned != upstream
    assert cleaned[0].startswith("Please act as an impartial judge following")


def test_judge_mt_bench_101_includes_reference_block_for_mr():
    eval_items = pd.DataFrame(
        {
            "instruction_index": [0],
            "dialogue_id": [1],
            "dialogue_uid": ["MR:1"],
            "task": ["MR"],
            "ability": ["reasoning"],
            "domain": ["adaptability"],
            "turn_index": [2],
            "golden_context": [[{"user": "q1", "bot": "a1"}]],
            "user_message": ["q2"],
            "reference_answer": ["ref answer"],
        }
    ).set_index("instruction_index")
    completions = pd.DataFrame(
        {"instruction_index": [0], "completion": ["<think>hidden</think>model answer"]}
    )
    scored = judge_mt_bench_101_single(
        judge_chat_model=DummyModel("Dummy/Rating: [[8]]"),
        eval_items=eval_items,
        completions=completions,
        evaluated_model="Dummy/model",
        use_tqdm=False,
        strip_thinking_before_judging=True,
    )
    user_prompt = scored.iloc[0]["user_prompt"]
    assert scored.iloc[0]["score"] == pytest.approx(8.0)
    assert "The reference solution is:" in user_prompt
    assert "ref answer" in user_prompt
    assert "hidden" not in user_prompt


def test_mt_bench_101_min_dialogue_and_pairwise():
    scored_a = pd.DataFrame(
        {
            "instruction_index": [0, 1, 2],
            "dialogue_uid": ["PI:1", "PI:1", "PI:2"],
            "dialogue_id": [1, 1, 2],
            "task": ["PI", "PI", "PI"],
            "ability": ["questioning", "questioning", "questioning"],
            "domain": ["interactivity", "interactivity", "interactivity"],
            "turn_index": [1, 2, 1],
            "score": [9.0, 2.0, 4.0],
        }
    )
    scored_b = scored_a.assign(score=[8.0, 1.0, 6.0])
    pairwise_turns = derive_mt_bench_101_pairwise_preferences(scored_a, scored_b)
    pairwise = aggregate_mt_bench_101_dialogues(pairwise_turns)
    absolute_scores = build_metric("mt_bench_101_absolute_score").calculate(pairwise)
    assert absolute_scores["model_A_score"] == pytest.approx(3.0)
    assert len(pairwise) == 2
    assert pairwise["score_A"].tolist() == [2.0, 4.0]
    assert pairwise["score_B"].tolist() == [1.0, 6.0]


def test_mt_bench_101_resolves_upstream_sampling_defaults():
    cfg = RunConfig(
        task="mt-bench-101",
        model={"name": "Dummy/a", "baseline": "Dummy/b"},
        judge={"model": "Dummy/judge"},
    )
    assert cfg.model.temperature == 0.0
    assert cfg.model.baseline_generation_kwargs()["temperature"] == 0.0
    assert cfg.model.max_out_tokens == 4096
    assert cfg.judge.temperature == 0.6
    assert cfg.judge.max_out_tokens == 4096

    overridden = RunConfig(
        task="mt-bench-101",
        model={
            "name": "Dummy/a",
            "baseline": "Dummy/b",
            "temperature": 0.2,
            "max_out_tokens": 128,
        },
        judge={
            "model": "Dummy/judge",
            "temperature": 0.1,
            "max_out_tokens": 64,
        },
    )
    assert overridden.model.temperature == 0.2
    assert overridden.model.max_out_tokens == 128
    assert overridden.judge.temperature == 0.1
    assert overridden.judge.max_out_tokens == 64


def test_run_mt_bench_101_dummy_reuses_cache(tmp_path, monkeypatch):
    eval_items = expand_mt_bench_101_records(
        [{"task": "PI", "id": 1, "history": [{"user": "q", "bot": "a"}]}]
    ).set_index("instruction_index")
    monkeypatch.setattr(runner, "load_instructions", lambda *_a, **_k: eval_items)
    store_root = tmp_path / "cache"
    cfg = RunConfig(
        task="mt-bench-101",
        model={"name": "Dummy/ok-a", "baseline": "Dummy/ok-b"},
        judge={"model": "Dummy/Rating: [[8]]"},
        generation={"n_instructions": 1},
        run={
            "result_folder": str(tmp_path),
            "store_root": str(store_root),
            "use_tqdm": False,
        },
    )
    task = get_packaged_task("mt-bench-101")
    prefs = run_mt_bench_101_benchmark(cfg, task)
    assert prefs.tolist() == pytest.approx([0.5])

    for kind, db_name, store_type in (
        ("completions", COMPLETION_DB_NAME, CompletionCache),
        ("judgements", JUDGEMENT_DB_NAME, JudgementCache),
    ):
        instruction_ids = []
        for db_path in (store_root / kind / "mt-bench-101").rglob(db_name):
            with store_type(db_path) as cache:
                instruction_ids += cache.query()["instruction_id"].tolist()
        assert len(instruction_ids) == 2
        assert all(re.fullmatch(r"PI:1:turn-\d+", i) for i in instruction_ids)

    def fail_if_materialized(*_args, **_kwargs):
        raise AssertionError("cache hit materialized a model")

    monkeypatch.setattr(models, "make_model", fail_if_materialized)
    assert run_mt_bench_101_benchmark(cfg, task).tolist() == prefs.tolist()
