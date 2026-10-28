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
        result_type=ParsedScore,
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


def test_executor_preserves_responses_missing_required_logprobs(caplog):
    parser = SimpleNamespace(
        name="requires-logprobs",
        result_type=ParsedScore,
        requires_top_logprobs=True,
        parse_result=lambda *_args, **_kwargs: pytest.fail("Missing parser input"),
    )
    outputs = [
        "M",
        AIMessage(
            content="m",
            response_metadata={"logprobs": {"content": [{"top_logprobs": []}]}},
        ),
        AIMessage(
            content="M",
            response_metadata={
                "logprobs": {
                    "content": [{"top_logprobs": [{"token": "x", "logprob": -0.1}]}]
                }
            },
        ),
        "score_A: 8\nscore_B: 7",
    ]
    requests = [
        JudgeRequest(str(i), "Q", parse)
        for i, parse in enumerate(
            [
                parser,
                parser,
                JUDGE_PARSERS["alpaca-eval-token"],
                JUDGE_PARSERS["score"],
            ]
        )
    ]

    results = execute_judging(
        SimpleNamespace(batch=lambda inputs, **_: outputs), requests
    )

    assert [r.response.text for r in results] == [
        "M",
        "m",
        "M",
        "score_A: 8\nscore_B: 7",
    ]
    assert [r.parsed for r in results[:3]] == [None, None, None]
    assert results[3].parsed.scores == {"A": 8.0, "B": 7.0}
    warnings = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "2" in warnings[0] and "logprobs" in warnings[0]


def test_executor_rejects_duplicate_ids_before_inference():
    request = JudgeRequest("same", "prompt", PairScore())
    with pytest.raises(ValueError, match="unique"):
        execute_judging(object(), [request, request])


def test_executor_empty_batch_does_not_need_a_model():
    assert execute_judging(object(), []) == []


def test_executor_rejects_result_that_breaks_parser_contract():
    parser = SimpleNamespace(
        name="broken-score",
        result_type=ParsedScore,
        requires_top_logprobs=False,
        parse_result=lambda text, **_: {"score": 8},
    )
    with pytest.raises(TypeError, match="broken-score.*ParsedScore"):
        execute_judging(FakeListLLM(responses=["8"]), [JudgeRequest("q", "Q", parser)])


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


@pytest.mark.parametrize("mode", ["raw", "parser", "resolved"])
def test_pairwise_annotation_can_parse_without_changing_rendered_inputs(mode):
    parse = None if mode == "raw" else JUDGE_PARSERS["arena-hard-verdict"]
    prompt_args = {
        "system_prompt": "Compare",
        "user_prompt_template": "{user_prompt}|{completion_A}|{completion_B}",
    }
    if mode == "resolved":
        prompt_args = {
            "prompt": replace(
                evaluate.resolve_judge_prompts(**prompt_args), parser=parse
            )
        }
    else:
        prompt_args["parse"] = parse
    annotation, failed = evaluate.annotate_battles(
        FakeListLLM(responses=["[[A>B]]", "no verdict"]),
        ["Question {x}", "Second question"],
        ["<think>private</think>ABCDE", "Answer A"],
        ["FGHIJ", "Answer B"],
        truncate_input_chars=3,
        strip_thinking_before_judging=True,
        **prompt_args,
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
    assert failed.instruction == "Second question"
    assert (failed.completion_A, failed.completion_B) == ("Answer A", "Answer B")
    assert failed.judge_completion == "no verdict"
    assert failed.parsed is None


@pytest.mark.parametrize("resolved", [False, True])
def test_pairwise_annotation_rejects_delegated_prompts_before_inference(resolved):
    prompt_args = (
        {"prompt": evaluate.resolve_judge_prompt(preset="fastchat-pairwise")}
        if resolved
        else {"prompt_preset": "fastchat-pairwise"}
    )
    with pytest.raises(ValueError, match="delegated"):
        evaluate.annotate_battles(object(), ["Q"], ["A"], ["B"], **prompt_args)


@pytest.mark.parametrize("positional", [False, True])
def test_swapping_moves_asymmetric_candidate_metadata(positional):
    import pandas as pd

    from judgearena.battles import swap_comparison_slots

    pairs = pd.DataFrame(
        {
            "model_a": ["a", "a"],
            "model_b": ["b", "b"],
            "completion_a": "A",
            "completion_b": "B",
            "checkpoint_a": ["saved", "other"],
            "lang": ["en", "fr"],
        }
    )
    pairs.index = ["x", "y"]
    mask = pd.Series([True, False], index=pairs.index)
    swapped = swap_comparison_slots(pairs, mask.to_numpy() if positional else mask)
    assert swapped["checkpoint_b"].iloc[0] == "saved"
    assert pd.isna(swapped["checkpoint_a"].iloc[0])
    assert swapped["checkpoint_a"].iloc[1] == "other"
    assert swapped["lang"].tolist() == ["en", "fr"]
    assert "checkpoint_b" not in pairs
    for invalid in (
        mask.reset_index(drop=True),
        mask.iloc[::-1],
        mask.astype(int),
        mask.to_numpy()[:1],
        pd.Series([True, pd.NA], index=pairs.index, dtype="boolean"),
    ):
        with pytest.raises(ValueError, match="mask"):
            swap_comparison_slots(pairs, invalid)


@pytest.mark.parametrize("conversations", [False, True])
def test_comparison_preflight_checks_all_selected_batches(conversations):
    import pandas as pd

    from judgearena.benchmarks.pairwise.judging import judge_comparisons
    from judgearena.prompts.registry import resolve_judge_prompt

    pairs = pd.DataFrame(
        {"model_a": "a", "model_b": "b", "instruction_id": "shared-source"},
        index=["first", "second", "invalid"],
    )
    if conversations:
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
        pairs = pairs.assign(
            messages_a=[messages] * 3,
            messages_b=[messages, messages, None],
            completion_a=[None, "A", None],
            completion_b=[None, "A", None],
        )
    else:
        pairs = pairs.assign(
            instruction="Q", completion_a="A", completion_b=["B", "B", None]
        )
    prompt = resolve_judge_prompt(preset="default")
    prompt = replace(
        prompt,
        user_prompt_template=prompt.user_prompt_template
        + "\nRaw: {completion_A_text}|{completion_B_text}",
    )
    options = dict(
        swap_mode="both",
        strip_thinking_before_judging=False,
        truncate_input_chars=None,
        use_tqdm=False,
    )
    unavailable = SimpleNamespace(
        batch=lambda *args, **kwargs: pytest.fail("Judging started before validation")
    )
    with pytest.raises(ValueError, match="messages|missing"):
        judge_comparisons(
            unavailable,
            pairs,
            [(prompt, pairs.index[:2]), (prompt, pairs.index[2:])],
            **options,
        )

    annotations = judge_comparisons(
        FakeListLLM(responses=["score_A: 8 score_B: 6"]),
        pairs,
        [(prompt, pairs.index[:2])],
        **options,
    )
    if conversations:
        assert annotations.judge_input.str.endswith("Raw: A|A").all()
    assert annotations.instruction_index.tolist() == ["first", "second"] * 2
    assert annotations.instruction_id.tolist() == ["shared-source"] * 4
    assert annotations.pref.iloc[:2].tolist() == pytest.approx(
        [1 - p for p in annotations.pref.iloc[2:]]
    )

    for ids in ([1, True, "invalid"], [1, "1", "invalid"]):
        ambiguous = pairs.set_axis(pd.Index(ids, dtype=object))
        with pytest.raises(ValueError, match="unique"):
            judge_comparisons(
                unavailable, ambiguous, [(prompt, ambiguous.index[:2])], **options
            )
