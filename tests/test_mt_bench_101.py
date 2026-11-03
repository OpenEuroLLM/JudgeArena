import hashlib
import json
import re
from dataclasses import replace
from functools import partial

import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM

import judgearena.benchmarks.evaluation as runner
import judgearena.datasets.mt_bench_101 as mt_bench_101
import judgearena.models as models
from judgearena.benchmarks.evaluation import run_evaluation
from judgearena.benchmarks.pairwise.scoring.metrics import score_outcomes
from judgearena.benchmarks.scoring import build_metric
from judgearena.cache.sqlite import (
    COMPLETION_DB_NAME,
    JUDGEMENT_DB_NAME,
    CompletionCache,
    JudgementCache,
)
from judgearena.config import RunConfig
from judgearena.datasets.data import CompletionData
from judgearena.datasets.mt_bench_101 import expand_mt_bench_101_records
from judgearena.judging import judge_single_answers
from judgearena.prompts.mt_bench_101 import (
    MTBench101ScoreParser,
    load_mt_bench_101_prompts,
    render_mt_bench_101_requests,
)
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


def test_download_mt_bench_101_uses_pinned_revision(tmp_path, monkeypatch):
    task = get_packaged_task("mt-bench-101")
    spec = task.spec.model_dump()
    spec["dataset"]["sources"]["benchmark"]["revision"] = "1" * 40
    custom = replace(task, spec=type(task.spec).model_validate(spec))
    calls = []

    def _urlretrieve(url, dest):
        calls.append(url)
        dest.write_text(
            json.dumps(
                {"task": "PI", "id": 9, "history": [{"user": url, "bot": "answer"}]}
            )
            + "\n"
        )

    monkeypatch.setattr(mt_bench_101, "urlretrieve", _urlretrieve)
    for selected in (task, custom, task):
        loaded = mt_bench_101.MTBench101Adapter(selected, tmp_path).load_instructions()
        assert (
            selected.spec.dataset.sources["benchmark"].revision
            in loaded.iloc[0]["instruction"]
        )
    assert len(calls) == 2
    assert "bc18b3e2c18c99164e11528f1a79c92083db5953" in calls[0]
    assert calls[0].endswith("data/subjective/mtbench101.jsonl")


def test_parse_mt_bench_101_rating():
    parse_mt_bench_101_rating = MTBench101ScoreParser()
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
        MTBench101ScoreParser(MT_BENCH_101_CLEAN_PROMPT_PRESET)("Rating: [6]") is None
    )


@pytest.mark.parametrize(
    ("preset", "expected"),
    [(MT_BENCH_101_PROMPT_PRESET, 2.0), (MT_BENCH_101_CLEAN_PROMPT_PRESET, 9.0)],
)
def test_mt_bench_101_judging_keeps_turn_alignment_with_missing_rating(
    preset, expected
):
    items = expand_mt_bench_101_records(
        [
            {
                "task": "MR",
                "id": 1,
                "history": [{"user": f"q{i}", "bot": f"ref{i}"} for i in range(1, 4)],
            }
        ]
    ).set_index("instruction_index")
    completions = pd.DataFrame(
        {
            "instruction_index": items.index,
            "completion": ["<think>hidden</think>answer1", "answer2", "answer3"],
        }
    ).iloc[::-1]
    outputs = ["<think>[[2]]</think> Rating: [[9]]", "no rating", "Rating: [[3]]"]
    scored, _ = judge_single_answers(
        FakeListLLM(responses=outputs),
        items.assign(
            completion=completions.set_index("instruction_index")["completion"],
            model="candidate",
            language=["tr", "en", "de"],
            judge_completion="source metadata",
            parsed="source metadata",
        ),
        partial(
            render_mt_bench_101_requests,
            prompt_preset=preset,
            strip_thinking_before_judging=True,
        ),
    )
    assert scored["language"].tolist() == ["tr", "en", "de"]
    assert scored.loc[0, "parsed"]["score"] == expected
    assert scored.loc[1, "parsed"] is None
    assert "ref1" in scored.loc[1, "judge_input"]
    assert scored["turn_index"].tolist() == [1, 2, 3]
    assert scored["model_completion"].tolist() == ["answer1", "answer2", "answer3"]
    assert scored["judge_completion"].tolist() == outputs
    assert scored.loc[[0, 2], "score"].tolist() == [expected, 3.0]
    assert pd.isna(scored.loc[1, "score"])
    assert "ref1" in scored.loc[0, "user_prompt"]
    assert "hidden" not in scored.loc[0, "user_prompt"]
    assert "ref1" in scored.loc[1, "user_prompt"]
    assert "answer1" not in scored.loc[1, "user_prompt"]


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


def test_mt_bench_101_min_dialogue_and_pairwise():
    scored_a = pd.DataFrame(
        {
            "instruction_index": [0, 1, 2, 3],
            "dialogue_uid": ["PI:1", "PI:1", "PI:2", "PI:3"],
            "dialogue_id": [1, 1, 2, 3],
            "task": ["PI"] * 4,
            "ability": ["questioning"] * 4,
            "domain": ["interactivity"] * 4,
            "turn_index": [1, 2, 1, 1],
            "score": [9.0, 2.0, 4.0, None],
        }
    )
    # Each model's minimum occurs on a different turn of the first dialogue.
    pairwise_turns = score_outcomes(
        scored_a.rename(columns={"score": "score_A"}).assign(
            score_B=[1.0, 8.0, 6.0, 5.0],
            model_a="a",
            model_b="b",
        )
    )
    pairwise = score_outcomes(pairwise_turns, minimum_by=("dialogue_uid", "task"))
    absolute_scores = build_metric(
        "absolute_score", parameters={"macro_by": "task"}
    ).calculate(pairwise)
    assert absolute_scores["model_A_score"] == pytest.approx(3.0)
    assert len(pairwise) == 3
    assert pairwise["score_A"].iloc[:2].tolist() == [2.0, 4.0]
    assert pairwise["score_B"].tolist() == [1.0, 6.0, 5.0]
    assert pairwise_turns["pref"].iloc[:3].tolist() == pytest.approx(
        [0.0831726965, 0.8581489351, 0.6456563062]
    )
    assert pairwise["pref"].iloc[:2].tolist() == pytest.approx(
        [0.4255574832, 0.6456563062]
    )
    assert pd.isna(pairwise_turns["pref"].iloc[3])
    assert pd.isna(pairwise["pref"].iloc[2])


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


@pytest.mark.parametrize("with_preloaded", [False, True])
def test_run_mt_bench_101_dummy_reuses_cache(tmp_path, monkeypatch, with_preloaded):
    eval_items = expand_mt_bench_101_records(
        [
            {
                "task": "PI",
                "id": 1,
                "history": [
                    {"user": "q{1}", "bot": "golden"},
                    {"user": "follow-up", "bot": "a2"},
                ],
            },
            {"task": "PI", "id": 2, "history": [{"user": "excluded", "bot": "a"}]},
        ]
    ).set_index("instruction_index")
    outputs = (
        pd.DataFrame(
            [
                (1, "Dummy/ok-b", "saved-b"),
                (0, "Dummy/ok-a", "saved-a"),
                (0, "Dummy/ok-b", ""),
            ],
            columns=["instruction_index", "model", "completion"],
        )
        if with_preloaded
        else None
    )
    saved_history = [
        {"role": "user", "content": "saved question"},
        {"role": "assistant", "content": "saved-b"},
    ]
    if outputs is not None:
        outputs["messages"] = [saved_history, None, None]
    data = CompletionData(eval_items, outputs)
    monkeypatch.setattr(runner, "load_task_data", lambda *_a, **_k: data)
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
    prefs = run_evaluation(cfg, task)
    assert prefs.tolist() == pytest.approx([0.5])
    report = json.loads(next(tmp_path.glob("*/results-*.json")).read_text())
    variants = report["metadata"]["judge_prompts"]
    assert len(variants) == 1
    assert variants[0]["judge_parser"] == "mt-bench-101-score"
    assert "{history}" in variants[0]["user_prompt_template"]
    assert "follow-up" not in variants[0]["user_prompt_template"]

    for kind, db_name, store_type in (
        ("completions", COMPLETION_DB_NAME, CompletionCache),
        ("judgements", JUDGEMENT_DB_NAME, JudgementCache),
    ):
        instruction_ids = []
        for db_path in (store_root / kind / "mt-bench-101").rglob(db_name):
            with store_type(db_path) as cache:
                instruction_ids += cache.query()["instruction_id"].tolist()
        assert len(instruction_ids) == (
            1 if kind == "completions" and with_preloaded else 4
        )
        assert all(re.fullmatch(r"PI:1:turn-\d+", i) for i in instruction_ids)

    if with_preloaded:
        annotations = pd.read_csv(
            next(tmp_path.glob("*/*-annotations.csv")), keep_default_na=False
        )
        assert annotations["model_completion"].tolist() == [
            "saved-a",
            "ok-a",
            "",
            "saved-b",
        ]
        assert annotations["messages"].ne("").all()
        assert annotations["messages"].iloc[:2].map(json.loads).tolist() == [
            eval_items.loc[i, "messages"] + [{"role": "assistant", "content": answer}]
            for i, answer in [(0, "saved-a"), (1, "ok-a")]
        ]
        assert annotations["messages"].iloc[2:].map(json.loads).tolist() == [
            [{"role": "user", "content": "q{1}"}, {"role": "assistant", "content": ""}],
            saved_history,
        ]
        assert eval_items.loc[0, "messages"] == [{"role": "user", "content": "q{1}"}]
        with CompletionCache(
            next((store_root / "completions").rglob(COMPLETION_DB_NAME))
        ) as cache:
            cached = cache.query()
        assert cached["instruction_id"].tolist() == ["PI:1:turn-2"]
        assert json.loads(cached.iloc[0]["input_text"])["messages"] == [
            {"role": "user", "content": "q{1}"},
            {"role": "assistant", "content": "golden"},
            {"role": "user", "content": "follow-up"},
        ]

    def fail_if_materialized(*_args, **_kwargs):
        raise AssertionError("cache hit materialized a model")

    monkeypatch.setattr(models, "make_model", fail_if_materialized)
    assert run_evaluation(cfg, task).tolist() == prefs.tolist()
