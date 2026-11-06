import json
from io import BytesIO

import pandas as pd
import pytest

import judgearena.models as models
from judgearena.cache.inference import JudgementInferenceCache
from judgearena.judging import JudgeRequest, execute_judging
from judgearena.prompts.registry import resolve_judge_prompt
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
        question="Is this safe?",
        endpoint="https://example.test/decisions",
        state=None,
        result=None,
    ):
        model = models.prepare_model(
            "OpenRouter/example/model",
            cache=cache,
            endpoint=endpoint,
            api_key_env="DECISION_KEY",
        )
        questions = {
            "safe": {"type": "noul", "instructions": question},
            "winner": {"type": "choice", "criteria": {"A": "First", "B": "Second"}},
            "quality": {"type": "score", "criteria": ["Poor", "Good"]},
        }
        prompt = resolve_judge_prompt(
            preset="decision",
            options={
                "questions": questions,
                "result": result,
                "judging_mode": "pairwise" if result else "single_answer",
            },
        )
        request = JudgeRequest(
            "1",
            {"state": state or "Hello", "questions": questions},
            prompt.parser,
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
    conversion = {"question": "winner", "values": {"A": 0, "B": 1}}
    assert judge(result={"preference": conversion}).parsed.preference == 0
    weighted = judge(result={"preference": {**conversion, "aggregation": "expected"}})
    assert weighted.parsed.preference == pytest.approx(0.2)
    assert weighted.parsed.details == response["answers"]
    for probabilities in ({"A": 1}, {"A": 0.2, "B": 0.2}):
        malformed = {"answers": {"winner": {"probabilities": probabilities}}}
        assert weighted.request.parser.parse_result(json.dumps(malformed)) is None
    assert len(calls) == 1  # Reparse the raw cache without credentials or HTTP.
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
        "ChatOpenAI/local",
        input_mode="decision",
        endpoint="http://localhost/decisions",
    )
    with pytest.raises(ValueError, match="requested answers"):
        model.invoke({"state": "Hello", "questions": {"safe": {"type": "noul"}}})


@pytest.mark.parametrize(
    "kind,output,invert,value,expected",
    [
        ("noul", "preference", True, 0.8, 0.2),
        ("noul", "preference", False, 0.8, 0.8),
        ("noul", "score", False, 0.8, 0.8),
        ("score", "score", False, 1.7, 1.7),
        ("noul", "score", False, 1.2, None),
        ("score", "score", False, float("nan"), None),
    ],
)
def test_decision_scalar_conversions(kind, output, invert, value, expected):
    options = {
        "questions": {"quality": {"type": kind}},
        "judging_mode": "pairwise" if output == "preference" else "single_answer",
        "result": {
            output: {
                "question": "quality",
                **({"invert": invert} if output == "preference" else {}),
            }
        },
    }
    parser = resolve_judge_prompt(preset="decision", options=options).parser
    answers = {"quality": {kind: value}, "extra": {"noul": 0.5}}
    parsed = parser.parse_result(json.dumps({"answers": answers}))
    if expected is None:
        assert parsed is None
    else:
        assert getattr(parsed, output) == pytest.approx(expected)
        assert parsed.details == answers
    assert parser.parse_result('{"answers": {"other": {}}}') is None


def test_decision_scores_use_shared_pairwise_conversion(monkeypatch, tmp_path):
    from judgearena.benchmarks.pairwise.judging import prepare_pairwise_judging
    from judgearena.config import RunConfig
    from judgearena.prompts.registry import resolve_run_judge_prompt

    cfg = RunConfig(
        task="alpaca-eval-ja",
        model={"name": "Dummy/a"},
        judge={
            "model": "ChatOpenAI/local",
            "prompt": {
                "preset": "decision",
                "options": {
                    "questions": {
                        "quality": {
                            "type": "score",
                            "criteria": ["Poor", "Good", "Excellent"],
                        }
                    },
                    "result": {"score": {"question": "quality"}},
                },
            },
        },
    )
    pairs = pd.DataFrame(
        [
            {
                "instruction": "Q",
                "model_a": "a",
                "model_b": "b",
                "completion_a": "A",
                "completion_b": "B",
            }
        ]
    )

    def post(request, **kwargs):
        state = json.loads(request.data)["state"]
        score = 2 if state["completion"] == "A" else 0
        return BytesIO(json.dumps({"answers": {"quality": {"score": score}}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", post)
    execute, _ = prepare_pairwise_judging(
        cfg,
        resolve_run_judge_prompt(cfg.resolved_task, cfg.judge),
        pairs,
        available_columns=pairs.columns,
    )
    model = models.prepare_model(
        "ChatOpenAI/local",
        endpoint="http://localhost/decisions",
        cache=JudgementInferenceCache(tmp_path, "scores"),
    )
    annotations, outcomes, _ = execute(model, pairs)
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: pytest.fail("Cache miss")
    )
    replay, replay_outcomes, _ = execute(model, pairs)
    pd.testing.assert_frame_equal(annotations, replay)
    pd.testing.assert_frame_equal(outcomes, replay_outcomes)
    from judgearena.artifacts import write_annotations

    write_annotations(annotations, tmp_path / "annotations.csv")
    saved = pd.read_csv(tmp_path / "annotations.csv")
    assert json.loads(saved.iloc[0].judge_input)["state"]["completion"] == "A"
    assert json.loads(saved.iloc[0].parsed)["details"]["quality"]["score"] == 2
    from judgearena.prompts.parsing import PairScore

    expected = PairScore(
        temperature=cfg.resolved_task.spec.protocol.judge.score_temperature
    ).preference_from_scores(2, 0)
    assert outcomes["pref"].tolist() == pytest.approx([expected])
    assert annotations["score"].tolist() == [2, 0]


def test_decision_state_and_result_contracts():
    from judgearena.prompts.registry import render_inputs

    options = {"questions": {"safe": {"type": "noul"}}, "state": {"language": "lang"}}
    prompt = resolve_judge_prompt(preset="decision", options=options)
    assert render_inputs(prompt, [{"lang": "de"}])[0]["state"] == {"language": "de"}
    with pytest.raises(ValueError, match="fields"):
        render_inputs(prompt, [{}])
    pairwise = resolve_judge_prompt(
        preset="decision",
        options={
            "questions": options["questions"],
            "judging_mode": "pairwise",
            "result": {"preference": {"question": "safe"}},
        },
    )
    row = {"instruction": "Q", "completion_a": "A", "completion_b": "B"}
    assert render_inputs(pairwise, [row])[0]["state"] == row
    for extra in (
        {"judging_mode": "pairwise"},
        {"state": {}},
        {"result": {"score": {"question": "missing"}}},
    ):
        with pytest.raises(ValueError):
            resolve_judge_prompt(preset="decision", options={**options, **extra})


def test_decision_provider_routing():
    model = models.make_model("OpenRouter/typesafe/jev", input_mode="decision")
    assert model.endpoint == "https://openrouter.ai/api/alpha/decisions"
    assert model.api_key_env == "OPENROUTER_API_KEY"
    with pytest.raises(ValueError, match="in-process"):
        models.make_model("VLLM/openjev/openjev", input_mode="decision")
