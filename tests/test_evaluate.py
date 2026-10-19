from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import AIMessage

import judgearena.evaluate as evaluate
from judgearena.cache.inference import JudgementInferenceCache
from judgearena.judging import JudgeRequest, execute_judging
from judgearena.models import prepare_model
from judgearena.prompts.parsing import JUDGE_PARSERS, PairScore, ParsedScore


def test_executor_keeps_mixed_parser_results_and_unparseable_responses_in_order():
    rating = SimpleNamespace(
        name="rating",
        requires_top_logprobs=False,
        parse_result=lambda text, **_: ParsedScore(score=float(text)),
    )
    outputs = [
        AIMessage(
            content="M",
            response_metadata={
                "logprobs": {
                    "content": [{"top_logprobs": [{"token": "M", "logprob": -0.5}]}]
                }
            },
        ),
        "8",
        "no verdict",
        "[[A>B]]",
    ]
    model = SimpleNamespace(batch=lambda inputs, **_: outputs)
    requests = [
        JudgeRequest(str(i), f"prompt {i}", parser)
        for i, parser in enumerate(
            [
                JUDGE_PARSERS["alpaca-eval-token"],
                rating,
                JUDGE_PARSERS["arena-hard-verdict"],
                JUDGE_PARSERS["arena-hard-verdict"],
            ]
        )
    ]

    results = execute_judging(model, requests)

    assert [result.request.id for result in results] == ["0", "1", "2", "3"]
    assert results[0].parsed.preference == 1.0
    assert results[0].response.first_token_top_logprobs == {"M": -0.5}
    assert results[1].parsed == ParsedScore(score=8)
    assert results[2].parsed is None
    assert results[2].response.text == "no verdict"
    assert results[3].parsed.preference == 0.25


def test_executor_reparses_cached_output_without_loading_the_model(tmp_path):
    cache = JudgementInferenceCache(tmp_path, "example")
    model_name = "Dummy/score_A: 6\nscore_B: 8"
    request = JudgeRequest(
        "q1",
        "Compare these responses",
        PairScore(temperature=0.5),
        cache_metadata={"instruction_id": "q1", "model_a": "a", "model_b": "b"},
    )
    (first,) = execute_judging(prepare_model(model_name, cache=cache), [request])
    cached_model = prepare_model(model_name, cache=cache)

    def unavailable():
        raise AssertionError("A cached judgment should not load the model")

    cached_model.factory = unavailable
    (second,) = execute_judging(
        cached_model, [replace(request, parser=PairScore(temperature=1))]
    )

    assert first.parsed.preference == pytest.approx(0.7310585786)
    assert second.parsed.preference == pytest.approx(0.8807970780)
    assert second.response.text == "score_A: 6\nscore_B: 8"


def test_executor_rejects_duplicate_ids_before_inference():
    request = JudgeRequest("same", "prompt", PairScore())
    with pytest.raises(ValueError, match="unique"):
        execute_judging(object(), [request, request])


def test_executor_empty_batch_does_not_need_a_model():
    assert execute_judging(object(), []) == []


def test_executor_rejects_partial_cache_metadata_before_inference():
    request = JudgeRequest("one", "prompt", PairScore())
    with pytest.raises(ValueError, match="metadata"):
        execute_judging(
            object(),
            [
                request,
                replace(
                    request,
                    id="two",
                    cache_metadata={
                        "instruction_id": "two",
                        "model_a": "a",
                        "model_b": None,
                    },
                ),
            ],
        )


@pytest.mark.parametrize("parse", [None, JUDGE_PARSERS["arena-hard-verdict"]])
def test_pairwise_annotation_can_parse_without_changing_rendered_inputs(parse):
    (annotation,) = evaluate.annotate_battles(
        FakeListLLM(responses=["[[A>B]]"]),
        ["Question {x}"],
        ["<think>private</think>ABCDE"],
        ["FGHIJ"],
        system_prompt="Compare",
        user_prompt_template="{user_prompt}|{completion_A}|{completion_B}",
        truncate_input_chars=3,
        strip_thinking_before_judging=True,
        parse=parse,
    )
    assert (
        annotation.judge_input.to_string()
        == "System: Compare\nHuman: Question {x}|ABC|FGH"
    )
    assert (annotation.completion_A, annotation.completion_B) == ("ABCDE", "FGHIJ")
    assert annotation.judge_completion == "[[A>B]]"
    assert annotation.parsed == (
        None if parse is None else evaluate.ParsedPreference(0.25, label="A>B")
    )
