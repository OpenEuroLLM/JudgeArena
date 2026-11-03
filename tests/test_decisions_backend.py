import json
from io import BytesIO

import pytest
from langchain_core.prompts import ChatPromptTemplate

import judgearena.models as models
from judgearena.cache.inference import JudgementInferenceCache
from judgearena.judging import JudgeRequest, execute_judging
from judgearena.usage import track_usage


class AnswersParser:
    result_type = dict
    requires_top_logprobs = False

    def parse_result(self, text, **_):
        return json.loads(text)["answers"]


@pytest.mark.parametrize("use_tqdm", [False, True])
def test_decisions_judging_replay_and_question_identity(
    tmp_path, monkeypatch, use_tqdm
):
    calls = []
    response = {
        "answers": {"safe": {"type": "noul", "noul": 0.9}},
        "usage": {"input_tokens": 12, "output_tokens": 0, "cost": 0.001},
    }

    def post(request, **_):
        calls.append(json.loads(request.data))
        assert request.get_header("Authorization") == "Bearer test-key"
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("urllib.request.urlopen", post)
    monkeypatch.setenv("DECISION_KEY", "test-key")
    cache = JudgementInferenceCache(tmp_path, "decision-test")
    prompt = ChatPromptTemplate.from_template("Evaluate: {text}").invoke(
        {"text": "Hello"}
    )
    request = JudgeRequest(
        "1",
        prompt,
        AnswersParser(),
        {"instruction_id": "1", "model_a": "candidate", "model_b": None},
    )

    def judge(question="Is this safe?", endpoint="https://example.test/decisions"):
        model = models.prepare_model(
            "Decisions/example/model",
            cache=cache,
            endpoint=endpoint,
            api_key_env="DECISION_KEY",
            questions={"safe": {"type": "noul", "instructions": question}},
        )
        return execute_judging(model, [request], use_tqdm=use_tqdm)[0]

    first = judge()
    assert first.parsed == response["answers"]
    assert first.response.usage.output_tokens == 0
    assert first.response.usage.cost_usd == 0.001
    assert calls == [
        {
            "model": "example/model",
            "state": prompt.to_string(),
            "questions": {"safe": {"type": "noul", "instructions": "Is this safe?"}},
        }
    ]
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
    assert len(calls) == 3


@pytest.mark.parametrize("response", [[], {}, {"answers": {"other": {"noul": 0.5}}}])
def test_decisions_reject_missing_answers(monkeypatch, response):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: BytesIO(json.dumps(response).encode())
    )
    model = models.make_model(
        "Decisions/local",
        endpoint="http://localhost/decisions",
        questions={"safe": {"type": "noul"}},
    )
    with pytest.raises(ValueError, match="requested answers"):
        model.invoke("Hello")
