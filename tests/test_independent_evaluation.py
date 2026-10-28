import json
from dataclasses import replace

import pandas as pd
import pytest

from judgearena.benchmarks.runner import run_benchmark
from judgearena.benchmarks.scoring import build_metric
from judgearena.config import RunConfig
from judgearena.datasets.arena_hard import ArenaHardAdapter
from judgearena.datasets.data import CompletionData
from judgearena.datasets.wildbench import WildBenchAdapter
from judgearena.prompts.parsing import JUDGE_PARSERS
from judgearena.prompts.registry import resolve_judge_prompt
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import TaskSpec


def _independent_task():
    task = get_packaged_task("arena-hard-v0.1")
    spec = task.model_dump()
    spec["protocol"] = {
        "runner": "single_answer",
        "generation": {"mode": "provided_messages"},
        "judge": {"default_prompt_preset": "single-score"},
        "scoring": {"metrics": [{"metric": "mean_score"}]},
    }
    return replace(task, spec=TaskSpec.model_validate(spec))


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", 1),
        ('"10"', 10),
        ("8.5", 8.5),
        ("0", None),
        ("11", None),
        ("true", None),
        ('"NaN"', None),
    ],
)
def test_single_score_parser_validates_numeric_range(value, expected):
    parsed = resolve_judge_prompt(preset="single-score").parser.parse_result(
        '{"strengths": "Clear", "score": ' + value + "}"
    )
    if expected is None:
        assert parsed is None
    else:
        assert parsed.score == expected
        assert parsed.details["strengths"] == "Clear"


def test_mean_score_preserves_scale_and_counts_missing_results():
    metric = build_metric("mean_score")
    assert metric.calculate(pd.DataFrame({"score": [-2, 4, None]})) == {
        "score": 1.0,
        "num_samples": 3,
        "num_missing": 1,
    }
    assert metric.calculate(pd.DataFrame({"score": [None]}))["score"] is None
    with pytest.raises(ValueError, match="finite"):
        metric.calculate(pd.DataFrame({"score": [float("inf")]}))


@pytest.mark.parametrize("value", ["true", '"NaN"', "Infinity", "null"])
def test_generic_score_parser_rejects_non_numeric_rewards(value):
    from judgearena.prompts.parsing import resolve_judge_parser

    assert (
        resolve_judge_parser("json-score").parse_result('{"score": ' + value + "}")
        is None
    )


@pytest.mark.parametrize("with_saved_history", [False, True])
def test_independent_evaluation_reuses_answers_and_preserves_results(
    monkeypatch, tmp_path, with_saved_history
):
    task = _independent_task()
    data = CompletionData.from_frames(
        pd.DataFrame(
            {
                "instruction_index": ["saved", "missing"],
                "instruction": ["Remember {42}", "Explain it"],
                "source_instruction_index": ["foreign-saved", "foreign-missing"],
                "messages": [
                    [
                        {"role": "user", "content": "Input question"},
                        {"role": "assistant", "content": "Input context"},
                        {"role": "user", "content": "Remember {42}"},
                    ],
                    [
                        {"role": "user", "content": "Remember {42}"},
                        {"role": "assistant", "content": "OK"},
                        {"role": "user", "content": "Explain it"},
                    ],
                ],
                "history": ["", "USER: Remember {42}\n\nASSISTANT: OK\n\n"],
                "checklist": ["- Relevant", "- Clear"],
                "category": ["Reasoning", "Reasoning"],
            }
        ),
        pd.DataFrame(
            {
                "instruction_index": ["saved"],
                "model": ["Dummy/generated"],
                "completion": ["stored answer"],
                "checkpoint": ["saved-v1"],
                "messages": [
                    [
                        {"role": "user", "content": "Earlier question"},
                        {"role": "assistant", "content": "Earlier answer"},
                        {"role": "user", "content": "Remember {42}"},
                        {"role": "assistant", "content": "stored answer"},
                    ]
                ],
            }
        ),
    )
    if not with_saved_history:
        data = replace(data, model_outputs=data.model_outputs.drop(columns="messages"))
    monkeypatch.setattr(ArenaHardAdapter, "load_data", lambda self: data)

    class Parser:
        name = "test-correctness"
        result_type = dict
        requires_top_logprobs = False

        def parse_result(self, text, **kwargs):
            return json.loads(text)

    monkeypatch.setitem(JUDGE_PARSERS, Parser.name, Parser())
    system, user = tmp_path / "system.txt", tmp_path / "user.txt"
    system.write_text("Use {checklist}")
    user.write_text(
        "Checkpoint: {checkpoint}\nQ: {user_prompt}\nA: {completion}\nRaw: {completion_text}"
    )
    judge = {
        "model": 'Dummy/{"correct": true, "criteria": {"clarity": "pass"}, "score": 0.8}',
        "prompt": {
            "system_file": str(system),
            "user_file": str(user),
            "parser": Parser.name,
        },
    }

    first = None
    for run in ("live", "replay"):
        cfg = RunConfig(
            task=task.task,
            resolved_task=task,
            model={"name": "Dummy/generated"},
            judge=judge,
            run={
                "store_root": str(tmp_path / "cache"),
                "result_folder": str(tmp_path / run),
                "use_tqdm": False,
                "no_log_file": True,
            },
        )
        annotations = run_benchmark(cfg)
        assert annotations.set_index("instruction_index").completion.to_dict() == {
            "missing": "generated",
            "saved": "stored answer",
        }
        saved = annotations.set_index("instruction_index").loc["saved"]
        assert saved.judge_input.endswith("Raw: stored answer")
        context = "Earlier answer" if with_saved_history else "Input context"
        assert context in saved.judge_input
        assert "saved-v1" in saved.judge_input
        assert saved.messages[-1]["content"] == "stored answer"
        assert annotations.category.tolist() == ["Reasoning", "Reasoning"]
        assert (
            "Remember {42}"
            in annotations.set_index("instruction_index").loc["saved", "judge_input"]
        )
        assert annotations.parsed.iloc[0] == {
            "correct": True,
            "criteria": {"clarity": "pass"},
            "score": 0.8,
        }
        assert annotations.score.tolist() == [0.8, 0.8]
        metadata = json.loads(
            next((tmp_path / run).rglob("run-metadata.v1.json")).read_text()
        )
        if first is None:
            assert metadata["usage"]["by_stage"]["generation"]["requests"] == 1
            assert metadata["usage"]["by_stage"]["judging"]["requests"] == 2
            first = annotations
        else:
            pd.testing.assert_frame_equal(first, annotations)
            assert metadata["usage"]["total"]["requests"] == 0
        report = json.loads(next((tmp_path / run).rglob("results-*.json")).read_text())
        assert report["metrics"]["mean_score"]["score"] == 0.8
        exported = pd.read_csv(next((tmp_path / run).rglob("*-annotations.csv")))
        assert json.loads(exported.parsed.iloc[0]) == annotations.parsed.iloc[0]

    # A complete saved conversation needs no generation history or turn plan.
    saved_only = CompletionData(
        data.instructions.loc[["saved"]].drop(
            columns=["messages"] if with_saved_history else []
        ),
        data.model_outputs,
    )
    with monkeypatch.context() as patch:
        patch.setattr(ArenaHardAdapter, "load_data", lambda self: saved_only)
        replay = run_benchmark(cfg)
        assert replay.messages.iloc[0] == saved.messages

    spec = task.model_dump()
    spec["protocol"].update(
        runner="pairwise",
        baseline={"strategy": "runtime_required"},
        judge={"default_prompt_preset": "single-score"},
        scoring={"metrics": [{"metric": "pairwise_win_rate"}]},
    )
    pairwise_task = replace(task, spec=TaskSpec.model_validate(spec))
    prefs = run_benchmark(
        RunConfig(
            task=task.task,
            resolved_task=pairwise_task,
            model={"name": "Dummy/generated", "baseline": "Dummy/reference"},
            judge={"model": 'Dummy/{"score": 8}'},
            run={
                "result_folder": str(tmp_path / "pairwise"),
                "store_root": str(tmp_path / "cache"),
                "use_tqdm": False,
            },
        )
    )
    assert prefs.tolist() == [0.5, 0.5]

    monkeypatch.setattr(WildBenchAdapter, "load_data", lambda self: data)
    score_dir = tmp_path / "wildbench-score"
    scored = run_benchmark(
        RunConfig(
            task="wildbench-score",
            model={"name": "Dummy/generated"},
            judge={"model": 'Dummy/{"score": 8}'},
            run={
                "result_folder": str(score_dir),
                "store_root": str(tmp_path / "cache"),
                "use_tqdm": False,
            },
        )
    )
    assert scored.score.tolist() == [8, 8]
    report = json.loads(next(score_dir.rglob("results-*.json")).read_text())
    assert report["metrics"]["wildbench_score"]["score"] == 6


def test_independent_rejects_pairwise_settings():
    task = _independent_task()
    for overrides in (
        {
            "model": {"name": "Dummy/x", "baseline": "Dummy/y"},
            "judge": {"model": "Dummy/x"},
        },
        {
            "model": {"name": "Dummy/x"},
            "judge": {"model": "Dummy/x", "swap_mode": "both"},
        },
    ):
        with pytest.raises(ValueError, match="single.answer|independent"):
            RunConfig(task=task.task, resolved_task=task, **overrides)


@pytest.mark.parametrize("value", ["1e2", "5e-1", "8garbage"])
def test_score_fallback_does_not_truncate_invalid_numbers(value):
    parser = resolve_judge_prompt(preset="single-score").parser
    assert parser.parse_result('```json\n{"score": ' + value + "}\n```") is None


def test_independent_empty_selection_fails_before_model_work(monkeypatch, tmp_path):
    data = CompletionData.from_frames(
        pd.DataFrame({"instruction_index": ["q"], "instruction": ["hi"]})
    )
    monkeypatch.setattr(ArenaHardAdapter, "load_data", lambda self: data)
    task = _independent_task()
    cfg = RunConfig(
        task=task.task,
        resolved_task=task,
        model={"name": "Dummy/x"},
        judge={"model": "Dummy/x"},
        generation={"n_instructions": 0},
        run={"result_folder": str(tmp_path)},
    )
    with pytest.raises(ValueError, match="No instructions"):
        run_benchmark(cfg)
    assert not list(tmp_path.iterdir())


def test_answer_samples_align_by_id_and_preserve_metadata():
    from judgearena.judging import answer_samples

    instructions = pd.DataFrame(
        {
            "instruction": ["Q8", "Q3"],
            "instruction_id": "shared-source",
            "history": ["H8", "H3"],
            "completion": "stale",
            "model": "stale",
        },
        index=pd.Index([8, 3], name="sample_id"),
    )
    answers = pd.DataFrame(
        {
            "completion": ["A3", "A8"],
            "model": ["model3", "model8"],
            "checkpoint": [3, 8],
        },
        index=[3, 8],
    )
    samples = answer_samples(instructions, answers)
    pd.testing.assert_frame_equal(
        samples,
        instructions.drop(columns=["completion", "model"]).join(answers),
    )
    assert instructions.completion.eq("stale").all()
    with pytest.raises(ValueError, match="overlap"):
        answer_samples(instructions.assign(checkpoint="question metadata"), answers)
    with pytest.raises(ValueError, match="missing"):
        answer_samples(instructions, answers.iloc[:1])
    with pytest.raises(ValueError, match="unique"):
        answer_samples(instructions, answers.set_axis([3, 3]))


def test_independent_pairs_validate_both_answers_before_judging():
    from judgearena.benchmarks.pairwise.judging import judge_independent_comparisons

    messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    comparisons = pd.DataFrame(
        {
            "instruction": ["Q"],
            "model_a": "a",
            "model_b": "b",
            "completion_a": "A",
            "completion_b": None,
            "messages_a": [messages],
            "messages_b": [messages],
        }
    )
    with pytest.raises(ValueError, match="missing"):
        judge_independent_comparisons(
            object(),
            comparisons,
            judge=lambda *args: pytest.fail("Judging started before validating B"),
            prompt=resolve_judge_prompt(preset="single-score"),
        )


def test_wildbench_score_reports_missing_and_normalized_endpoints():
    metric = build_metric("wildbench_score")
    for score, expected in [(1, -8), (5, 0), (10, 10)]:
        result = metric.calculate(pd.DataFrame({"score": [score, None]}))
        assert result["raw_mean"] == score
        assert result["score"] == expected
        assert result["num_missing"] == 1
    assert metric.calculate(pd.DataFrame({"score": [None]}))["score"] is None


@pytest.mark.parametrize("fenced", [False, True])
def test_json_scores_follow_analysis_code_and_preserve_nested_evidence(fenced):
    from judgearena.prompts.parsing import CriteriaScoreParser, ScoreParser

    verdict = '{"score": 8, "criteria": {"grammar": 8}, "details": {"score": 2}}'
    text = 'Example:\n```python\nprint({"score": 1})\n```\nFinal verdict:\n' + (
        f"```json\n{verdict}\n```" if fenced else verdict
    )
    for parser in (ScoreParser(), CriteriaScoreParser(["grammar"])):
        parsed = parser.parse_result(text)
        assert parsed.score == 8
        assert parsed.details["criteria"] == {"grammar": 8}
        assert parsed.details["details"] == {"score": 2}


@pytest.mark.parametrize("high,temperature", [(1, 0.3), (100, 0.003)])
def test_independent_comparison_temperature_matches_score_units(high, temperature):
    from langchain_core.language_models.fake import FakeListLLM

    from judgearena.benchmarks.pairwise.judging import prepare_pairwise_judging

    task = get_packaged_task("alpaca-eval")
    spec = task.spec.model_dump()
    spec["protocol"]["judge"].update(
        default_prompt_preset="single-score", score_temperature=temperature
    )
    task = replace(task, spec=TaskSpec.model_validate(spec))
    cfg = RunConfig(
        task=task.task,
        resolved_task=task,
        model={"name": "Dummy/a"},
        judge={"model": "Dummy/j"},
    )
    prompt = replace(
        resolve_judge_prompt(preset="single-score"),
        parser=JUDGE_PARSERS["json-score"],
        user_prompt_template="History: {input_messages} Answer: {completion}",
    )
    pairs = pd.DataFrame(
        {
            "instruction": ["Q"],
            "model_a": ["a"],
            "model_b": ["b"],
            "completion_a": ["A"],
            "completion_b": ["B"],
            "messages": [[{"role": "user", "content": "Q"}]],
        }
    )
    execute, _ = prepare_pairwise_judging(
        cfg, prompt, pairs, available_columns=pairs.columns
    )
    annotations, outcomes, _ = execute(
        FakeListLLM(responses=['{"score": 0}', json.dumps({"score": high})]), pairs
    )
    assert outcomes.pref.tolist() == pytest.approx([0.5744425168])
    assert annotations.turn_preference.tolist() == pytest.approx([0.5744425168] * 2)


@pytest.mark.parametrize("independent", [True, False])
def test_lazy_outputs_validate_prompt_before_generation(
    monkeypatch, tmp_path, independent
):
    import judgearena.benchmarks.pairwise.comparisons as preparation

    task = _independent_task() if independent else get_packaged_task("alpaca-eval")
    loaded = []
    data = CompletionData(
        pd.DataFrame(
            {"instruction": ["Q"], "messages": [[{"role": "user", "content": "Q"}]]}
        ),
        model_output_loader=lambda model: loaded.append(model),
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Rate {unknown_metadata}: {completion}")
    cfg = RunConfig(
        task=task.task,
        resolved_task=task,
        model={"name": "Dummy/a", "baseline": None if independent else "Dummy/b"},
        judge={
            "model": "Dummy/j",
            "prompt": {"user_file": str(prompt), "parser": "json-score"},
        },
        run={
            "result_folder": str(tmp_path),
            "store_root": str(tmp_path),
            "no_log_file": True,
        },
    )
    monkeypatch.setattr(
        preparation,
        "load_or_generate_completions",
        lambda *args, **kwargs: pytest.fail("Generation must not run"),
    )
    with pytest.raises(ValueError, match="unknown_metadata"):
        run_benchmark(cfg, data=data)
    assert loaded == (["Dummy/a"] if independent else ["Dummy/a", "Dummy/b"])
    assert data.model_output_loader is not None


@pytest.mark.parametrize("mode", ["user", "system", "criteria", "custom"])
def test_saved_run_replays_prompt_snapshot_after_original_changes(
    tmp_path, mode, monkeypatch
):
    from dataclasses import replace

    from judgearena.config import build_run_config, load_config
    from judgearena.prompts.criteria import configure_criteria_prompt
    from judgearena.prompts.registry import PRESETS

    def configure(prompt, *, rubric):
        return configure_criteria_prompt(prompt, criteria_file=rubric)

    monkeypatch.setitem(
        PRESETS,
        "custom-rubric",
        replace(
            PRESETS["criteria-score"],
            name="custom-rubric",
            configure=configure,
            option_files=("rubric",),
        ),
    )
    criteria_preset = "custom-rubric" if mode == "custom" else "criteria-score"
    option_key = "rubric" if mode == "custom" else "criteria_file"
    user = tmp_path / "rubric.txt"
    system = tmp_path / "system.txt"
    user.write_text('Rate {completion}. Return {{"score": number}}')
    system.write_text("Judge grammar only.")
    criteria = tmp_path / "rubric.yaml"
    criteria.write_text(
        "criteria: [{name: grammar, description: Grammatical correctness}]\n"
    )
    cfg = RunConfig(
        evaluation_name="snapshot",
        protocol={
            "runner": "single_answer",
            "generation": {"mode": "single_turn_chat"},
            "judge": {
                "default_prompt_preset": "criteria-score"
                if mode in ("criteria", "custom")
                else "single-score"
            },
        },
        model={"name": "Dummy/answer"},
        judge={
            "model": 'Dummy/{"score":8,"criteria":{"grammar":8}}',
            **(
                {
                    "prompt": {
                        "preset": criteria_preset,
                        "options": {option_key: criteria},
                    }
                }
                if mode in ("criteria", "custom")
                else {
                    "prompt": {
                        "user_file": user,
                        "system_file": system if mode == "system" else None,
                        "parser": "json-score",
                    }
                }
            ),
        },
        run={
            "result_folder": str(tmp_path / "results"),
            "store_root": str(tmp_path / "cache"),
            "no_log_file": True,
            "use_tqdm": False,
        },
    )
    data = CompletionData(pd.DataFrame({"instruction": ["Q"]}))
    first = run_benchmark(cfg, data=data)
    saved = next((tmp_path / "results").rglob("config.yaml"))
    user.unlink()
    system.unlink()
    criteria.unlink()
    moved = tmp_path / "moved-run"
    saved.parent.rename(moved)
    saved = moved / "config.yaml"
    restored = (
        build_run_config(["--config_path", str(saved)])
        if mode != "user"
        else load_config(saved)
    )
    replay = run_benchmark(restored, data=data)
    assert replay.judge_input.tolist() == first.judge_input.tolist()
    assert replay.prompt_preset.tolist() == first.prompt_preset.tolist()
    metadata = [
        json.loads(p.read_text()) for p in tmp_path.rglob("run-metadata.v1.json")
    ]
    assert sorted(m["usage"]["total"]["requests"] for m in metadata) == [0, 2]
    assert (
        cfg.judge.prompt.options[option_key] == criteria
        if mode in ("criteria", "custom")
        else cfg.judge.prompt.user_file == user
    )


def test_public_api_runs_custom_file_parser(tmp_path, monkeypatch):
    import judgearena.prompts.parsing as parsing
    from judgearena import (
        CompletionData,
        ParsedScore,
        RunConfig,
        register_parser,
        run_benchmark,
    )

    class GrammarParser:
        name = "grammar"
        result_type = ParsedScore
        requires_top_logprobs = False

        def parse_result(self, text, *, top_logprobs=None):
            return ParsedScore(float(text))

    monkeypatch.setattr(parsing, "JUDGE_PARSERS", dict(parsing.JUDGE_PARSERS))
    parser = register_parser("grammar", GrammarParser())
    for name, value in [("", parser), ("grammar", parser), ("broken", object())]:
        with pytest.raises(ValueError):
            register_parser(name, value)
    rubric = tmp_path / "grammar.txt"
    rubric.write_text("Rate grammar from 1 to 10: {completion}. Return a number.")
    cfg = RunConfig(
        evaluation_name="grammar",
        protocol={
            "runner": "single_answer",
            "generation": {"mode": "single_turn_chat"},
            "judge": {"default_prompt_preset": "single-score"},
        },
        model={"name": "saved"},
        judge={
            "model": "Dummy/8",
            "prompt": {"user_file": rubric, "parser": "grammar"},
        },
        run={"result_folder": str(tmp_path), "store_root": None, "no_log_file": True},
    )
    data = CompletionData(
        pd.DataFrame({"instruction": ["Say hello"]}),
        pd.DataFrame(
            {"instruction_index": [0], "model": ["saved"], "completion": ["Hello."]}
        ),
    )
    assert run_benchmark(cfg, data=data).score.tolist() == [8]
