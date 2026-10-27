import json
from dataclasses import replace

import pandas as pd
import pytest

import judgearena.benchmarks.pairwise.runner as mt_bench_runner
import judgearena.datasets.mt_bench as mt_bench
import judgearena.utils as judgearena_utils
from judgearena.config import RunConfig
from judgearena.datasets.data import CompletionData
from judgearena.datasets.registry import DatasetAdapter
from judgearena.prompts.registry import FASTCHAT_PAIRWISE_PROMPT_PRESET
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import TaskSelection


def test_mt_bench_answers_use_resolved_task_source_and_cache(monkeypatch, tmp_path):
    original = get_packaged_task("mt-bench")
    spec = original.spec.model_dump()
    spec["dataset"]["sources"]["benchmark"].update(
        repo_id="custom/mt-bench", revision="1234567890123456789012345678901234567890"
    )
    custom = replace(original, spec=type(original.spec).model_validate(spec))
    monkeypatch.setattr(judgearena_utils, "data_root", tmp_path)
    monkeypatch.setattr(
        mt_bench.MTBenchAdapter, "_download_references", lambda self: reference
    )
    reference = tmp_path / "reference.jsonl"
    reference.write_text('{"question_id": 7, "choices": [{"turns": ["R1", "R2"]}]}\n')
    downloads = []

    def snapshot(**kwargs):
        downloads.append(kwargs)
        path = kwargs["local_dir"] / kwargs["allow_patterns"][0]
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = f"{kwargs['repo_id']}@{kwargs['revision']}"
        if path.name == "question.jsonl":
            record = {
                "question_id": 7,
                "category": "math",
                "turns": [stamp, "Q2", "Q3"],
            }
        else:
            record = {"question_id": 7, "choices": [{"turns": [stamp, "A2", "A3"]}]}
        path.write_text(json.dumps(record) + "\n")

    monkeypatch.setattr(mt_bench, "snapshot_download", snapshot)
    for task in (original, custom, original):
        data = mt_bench_runner.load_task_data(task)
        source = task.spec.dataset.sources["benchmark"]
        expected = f"{source.repo_id}@{source.revision}"
        assert data.instructions.loc[7, "user_turns"] == [expected, "Q2", "Q3"]
        assert data.instructions.loc[
            7, ["category", "reference_turn_1", "reference_turn_2"]
        ].tolist() == ["math", "R1", "R2"]
        messages = data.model_outputs_for("candidate").iloc[0]["messages"]
        assert [message["content"] for message in messages] == [
            expected,
            expected,
            "Q2",
            "A2",
            "Q3",
            "A3",
        ]
    assert len(downloads) == 4  # One question and answer file per source.


@pytest.fixture
def loaded_mt_instructions(monkeypatch):
    data = CompletionData(pd.DataFrame({"instruction": ["q"], "user_turns": [["q"]]}))
    monkeypatch.setattr(mt_bench_runner, "load_task_data", lambda _: data)


def test_run_mt_bench_rejects_random_before_preparation(
    monkeypatch, loaded_mt_instructions
):
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "Dummy/model"},
        judge={
            "model": "Dummy/judge",
            "prompt_preset": FASTCHAT_PAIRWISE_PROMPT_PRESET,
            "swap_mode": "random",
        },
    )

    def unexpected_preparation(*_args, **_kwargs):
        pytest.fail("Unsupported swap mode must fail before preparing the run")

    monkeypatch.setattr(
        mt_bench_runner, "prepare_run_directory", unexpected_preparation
    )

    with pytest.raises(ValueError, match="MT-Bench supports only.*got 'random'"):
        mt_bench_runner.run_pairwise(cfg, get_packaged_task("mt-bench"))


def test_mt_renderer_rejects_logprob_parser_before_preparation(
    monkeypatch, loaded_mt_instructions
):
    from judgearena.prompts.registry import PRESETS

    monkeypatch.setitem(
        PRESETS,
        "fastchat-logprob",
        replace(
            PRESETS["alpaca-eval"],
            name="fastchat-logprob",
            prepare_prompts=PRESETS["fastchat-pairwise"].prepare_prompts,
        ),
    )
    cfg = RunConfig(
        task="mt-bench",
        model={"name": "Dummy/model"},
        judge={
            "model": "Dummy/judge",
            "prompt_preset": "fastchat-logprob",
            "top_logprobs": 5,
        },
    )

    def unexpected_preparation(*_args, **_kwargs):
        pytest.fail("Unsupported parser must fail before preparing the run")

    monkeypatch.setattr(
        mt_bench_runner, "prepare_run_directory", unexpected_preparation
    )

    with pytest.raises(ValueError, match="MT-Bench does not support.*logprobs"):
        mt_bench_runner.run_pairwise(cfg, get_packaged_task("mt-bench"))


@pytest.mark.parametrize(
    "preset,response,preference,turns",
    [
        ("fastchat-pairwise", "[[A]]", 0.0, (1, 2)),
        ("mt-score", "score A: 10 score B: 0", 0.0474258732, (1, 2)),
        ("default", "score A: 10 score B: 0", 0.0474258732, (1, 2, 3)),
        ("criteria", "clarity: A=8, B=4\ncorrectness: A=6, B=8", 0.4255574832, (2,)),
    ],
)
def test_prompt_bundle_controls_custom_conversation_judging(
    tmp_path, monkeypatch, preset, response, preference, turns
):
    import judgearena.datasets.registry as registry
    from judgearena.benchmarks.runner import run_benchmark
    from judgearena.cache.sqlite import JUDGEMENT_DB_NAME, JudgementCache
    from judgearena.prompts.registry import PRESETS

    monkeypatch.setitem(
        PRESETS,
        "mt-score",
        replace(
            PRESETS["default"],
            name="mt-score",
            prepare_prompts=PRESETS["fastchat-pairwise"].prepare_prompts,
        ),
    )
    questions = pd.DataFrame(
        {
            "instruction_index": [9, 3],
            "instruction": ["Question {one}", "Skip"],
            "user_turns": [
                ["Question {one}", "Follow-up", "Final question"],
                ["Skip", "Skip again"],
            ],
            "domain": ["keep", "skip"],
            "category": "math",
            "reference_turn_1": "reference one",
            "reference_turn_2": "reference two",
        }
    )
    outputs = pd.DataFrame(
        [
            {
                "instruction_index": identity,
                "model": model,
                "output": f"{identity}:first {model}",
                "messages": [
                    {"role": role, "content": text}
                    for role, text in (
                        ("user", "Question {one}"),
                        ("assistant", f"{identity}:first {model}"),
                        ("user", "Follow-up"),
                        ("assistant", f"{identity}:second {model}"),
                        ("user", "Final question"),
                        ("assistant", f"{identity}:third {model}"),
                    )
                ],
            }
            for identity, model in [(3, "a"), (9, "b"), (9, "a"), (3, "b")]
        ]
    )
    monkeypatch.setattr(registry, "_DATASETS", registry._DATASETS.copy())

    @registry.register_dataset("custom_conversations")
    class CustomConversationsAdapter(DatasetAdapter):
        def load_data(self):
            return CompletionData.from_frames(
                questions, model_output_loader=lambda _: outputs
            )

    rubric = tmp_path / "criteria.yaml"
    rubric.write_text(
        "criteria:\n  - name: clarity\n    description: 'Check {\"answer\": 42}.'\n  - name: correctness\n    description: Be correct.\n"
    )
    task = get_packaged_task("mt-bench")
    spec = task.spec.model_dump()
    spec["dataset"] = {"adapter": "custom_conversations"}
    spec["task"] = "custom-conversations"
    spec["protocol"]["judge"] = {
        "default_prompt_preset": preset,
        "turns": turns,
        "prompt_options": {"reference_categories": ["math"]},
    }
    task = replace(
        task,
        spec=type(task.spec).model_validate(spec),
        selection=TaskSelection(selector="domain", name="selected", values=("keep",)),
    )
    cfg = RunConfig(
        task=task.task,
        resolved_task=task,
        model={"name": "a", "baseline": "b"},
        judge={
            "model": f"Dummy/{response}",
            "prompt_preset": preset,
            "swap_mode": "both",
            "criteria_file": rubric if preset == "criteria" else None,
        },
        run={
            "result_folder": str(tmp_path),
            "store_root": str(tmp_path / "cache"),
            "use_tqdm": False,
        },
    )
    raw_preferences = [preference] * len(turns) + [1 - preference] * len(turns)
    expected = [0.5] * len(turns) if preset == "fastchat-pairwise" else raw_preferences
    assert run_benchmark(cfg).tolist() == pytest.approx(expected)
    annotations = pd.read_csv(next(tmp_path.rglob("*-annotations.csv")))
    assert (
        annotations.instruction_index.tolist()
        == [f"9:turn-{turn}" for turn in turns] * 2
    )
    assert annotations.pref.tolist() == pytest.approx(raw_preferences)
    assert annotations.orientation.tolist() == ["direct"] * len(turns) + [
        "reversed"
    ] * len(turns)
    for row in annotations.itertuples():
        assert "3:first" not in row.judge_input and "3:second" not in row.judge_input
        assert "Question {one}" in row.judge_input
        assert ("reference one" in row.judge_input) is (
            preset in {"fastchat-pairwise", "mt-score"}
        )
        if row.turn == 2:
            assert all(
                text in row.judge_input
                for text in ("Follow-up", "9:second a", "9:second b")
            )
        if row.turn == 3:
            assert all(
                text in row.judge_input
                for text in ("Final question", "9:third a", "9:third b")
            )
        if preset == "criteria":
            assert 'Check {"answer": 42}.' in row.judge_input
            assert json.loads(row.parsed)["scores"] == {"A": 7, "B": 6}
            assert json.loads(row.parsed)["details"]["criteria"] == {
                "clarity": {"A": 8, "B": 4},
                "correctness": {"A": 6, "B": 8},
            }
    results = json.loads(next(tmp_path.rglob("results-*.json")).read_text())
    assert results["metrics"]["pairwise_win_rate"]["winrate"] == 0.5
    if preset == "fastchat-pairwise":
        assert results["metadata"]["num_inconsistent"] == len(turns)
    metadata = json.loads(next(tmp_path.rglob("run-metadata.v1.json")).read_text())
    assert metadata["judge_prompts"] == results["metadata"]["judge_prompts"]
    with JudgementCache(next(tmp_path.rglob(JUDGEMENT_DB_NAME))) as cache:
        stored = cache.query()
    assert set(stored.instruction_id) == {f"9:turn-{turn}" for turn in turns}
    assert set(stored.orientation) == {"direct", "reversed"}
    assert len(stored) == len(turns) * 2

    from judgearena.models import PreparedModel

    def unavailable(_self):
        pytest.fail("Repeated judging must replay the cache")

    monkeypatch.setattr(PreparedModel, "materialize", unavailable)
    assert run_benchmark(cfg).tolist() == pytest.approx(expected)
