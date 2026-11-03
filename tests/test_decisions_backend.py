import json
from io import BytesIO

import pytest

import judgearena.models as models
from judgearena.cache.inference import JudgementInferenceCache
from judgearena.judging import JudgeRequest, execute_judging
from judgearena.prompts.parsing import resolve_judge_parser
from judgearena.usage import track_usage


@pytest.mark.parametrize("use_tqdm", [False, True])
def test_decisions_judging_replay_and_question_identity(
    tmp_path, monkeypatch, use_tqdm
):
    calls = []
    response = {
        "answers": {
            "safe": {"type": "noul", "noul": 0.9},
            "winner": {
                "type": "choice",
                "choice": "A",
                "probabilities": {"A": 0.8, "B": 0.2},
                "confidence": 0.6,
            },
            "quality": {
                "type": "score",
                "score": 0.8,
                "probabilities": [0.2, 0.8],
                "confidence": 0.6,
            },
        },
        "usage": {"input_tokens": 12, "output_tokens": 0, "cost": 0.001},
    }

    def post(request, **_):
        calls.append(json.loads(request.data))
        assert request.get_header("Authorization") == "Bearer test-key"
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("urllib.request.urlopen", post)
    monkeypatch.setenv("DECISION_KEY", "test-key")
    cache = JudgementInferenceCache(tmp_path, "decision-test")

    def judge(
        question="Is this safe?", endpoint="https://example.test/decisions", state=None
    ):
        model = models.prepare_model(
            "Decisions/example/model",
            cache=cache,
            endpoint=endpoint,
            api_key_env="DECISION_KEY",
        )
        request = JudgeRequest(
            "1",
            {
                "state": state or {"answer": "Hello"},
                "questions": {
                    "safe": {"type": "noul", "instructions": question},
                    "winner": {
                        "type": "choice",
                        "instructions": "Which is better?",
                        "criteria": {"A": "First", "B": "Second"},
                    },
                    "quality": {
                        "type": "score",
                        "instructions": "Rate quality",
                        "criteria": ["Poor", "Good"],
                    },
                },
            },
            resolve_judge_parser("decision-answers"),
            {"instruction_id": "1", "model_a": "candidate", "model_b": None},
        )
        return execute_judging(model, [request], use_tqdm=use_tqdm)[0]

    first = judge()
    assert first.parsed == response["answers"]
    assert first.response.usage.output_tokens == 0
    assert first.response.usage.cost_usd == 0.001
    assert calls == [{"model": "example/model", **first.request.input}]
    monkeypatch.delenv("DECISION_KEY")
    with track_usage() as usage:
        replay = judge()
    assert replay.parsed == first.parsed
    assert replay.response.text == first.response.text
    assert not usage.snapshot().requests
    assert len(calls) == 1
    monkeypatch.setenv("DECISION_KEY", "test-key")
    judge("Is this helpful?")
    judge(endpoint="https://other.test/decisions")
    judge(state=[{"role": "user", "content": "Different"}])
    assert len(calls) == 4


@pytest.mark.parametrize("response", [[], {}, {"answers": {"other": {"noul": 0.5}}}])
def test_decisions_reject_missing_answers(monkeypatch, response):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: BytesIO(json.dumps(response).encode())
    )
    model = models.make_model(
        "Decisions/local",
        endpoint="http://localhost/decisions",
    )
    with pytest.raises(ValueError, match="requested answers"):
        model.invoke({"state": "Hello", "questions": {"safe": {"type": "noul"}}})


@pytest.mark.parametrize("pairwise", [False, True])
def test_decision_bundle_preparation_and_annotations(tmp_path, monkeypatch, pairwise):
    import pandas as pd

    from judgearena.artifacts import write_annotations
    from judgearena.evaluate import (
        render_comparison_requests,
        render_single_requests,
        validate_prompt_fields,
    )
    from judgearena.judging import annotation_frame
    from judgearena.prompts.parsing import ParsedPreference
    from judgearena.prompts.registry import (
        PRESETS,
        JudgePromptPreset,
        register_prompt,
        resolve_judge_prompt,
    )

    class PreferenceParser:
        requires_top_logprobs = False
        result_type = ParsedPreference

        def parse_result(self, text, **kwargs):
            answers = resolve_judge_parser("decision-answers").parse_result(text)
            return ParsedPreference(0.0, details=answers)

    fields = (
        ("user_prompt", "completion_A", "completion_B")
        if pairwise
        else ("user_prompt", "completion")
    )
    questions = {
        "criterion": {"type": "noul", "instructions": "Is the answer helpful?"}
    }
    monkeypatch.setattr("judgearena.prompts.registry.PRESETS", PRESETS.copy())
    register_prompt(
        JudgePromptPreset(
            name="structured-test",
            parser=PreferenceParser()
            if pairwise
            else resolve_judge_parser("decision-answers"),
            judging_mode="pairwise" if pairwise else "single_answer",
            required_fields=fields,
            render_input=lambda row: {
                "state": {key: row[key] for key in fields},
                "questions": questions,
            },
        )
    )
    prompt = resolve_judge_prompt(preset="structured-test")
    with pytest.raises(ValueError, match="dataset columns"):
        validate_prompt_fields(prompt, (), ())
    rows = pd.DataFrame(
        [
            {
                "instruction": "A {question}",
                "completion": "answer long",
                "completion_a": "answer long",
                "completion_b": "other long",
            }
        ],
        index=["id"],
    )
    if pairwise:
        requests = render_comparison_requests(rows, prompt, 6)[0]
    else:
        _, requests, _ = render_single_requests(
            rows,
            prompt=prompt,
            truncate_input_chars=6,
            strip_thinking_before_judging=False,
        )
    assert requests[0].input == {
        "state": dict(zip(fields, ["A {question}", "answer", "other "], strict=False)),
        "questions": questions,
    }
    response = {"answers": {"criterion": {"type": "noul", "noul": 0.9}}}
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: BytesIO(json.dumps(response).encode())
    )
    results = execute_judging(
        models.prepare_model("Decisions/local", endpoint="http://localhost/decisions"),
        requests,
    )
    path = tmp_path / "annotations.csv"
    write_annotations(annotation_frame(rows, results, [prompt]), path)
    saved = pd.read_csv(path).iloc[0]
    assert json.loads(saved.judge_input) == requests[0].input
    assert json.loads(saved.judge_completion) == response
