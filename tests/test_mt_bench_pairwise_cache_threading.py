from __future__ import annotations

from dataclasses import dataclass

import judgearena.mt_bench.pairwise_judging as pairwise_module
from judgearena.inference_cache import InferenceCache
from judgearena.mt_bench.pairwise_judging import (
    MTBenchJudgeItem,
    infer_pairwise_judgments_by_prompt_groups,
)


@dataclass(frozen=True)
class _Prompt:
    name: str
    system_prompt: str | None
    user_prompt_template: str
    multi_turn: bool
    ref_based: bool = False


def _item(
    *,
    question_id: object,
    category: str | None,
    turn: int,
    prompt: _Prompt,
    prompt_kwargs: dict[str, str],
) -> MTBenchJudgeItem:
    return MTBenchJudgeItem(
        question_id=question_id,
        category=category,
        turn=turn,
        prompt=prompt,
        prompt_kwargs=prompt_kwargs,
    )


def test_infer_pairwise_judgments_metadata_order_and_orientation(monkeypatch):
    captured: list[dict] = []

    def spy_do_inference(*, cache_meta, **kwargs):
        captured.append(
            {
                "cache_meta": cache_meta,
                "input_count": len(kwargs["inputs"]),
            }
        )
        return ["judgment"] * len(kwargs["inputs"])

    monkeypatch.setattr(pairwise_module, "do_inference", spy_do_inference)

    single_prompt = _Prompt(
        name="default-single",
        system_prompt=None,
        user_prompt_template="{question} {answer_a} {answer_b}",
        multi_turn=False,
    )
    multi_prompt = _Prompt(
        name="default-multi",
        system_prompt=None,
        user_prompt_template="{question_1} {answer_a_1}",
        multi_turn=True,
    )
    items = [
        _item(
            question_id=1,
            category="writing",
            turn=1,
            prompt=single_prompt,
            prompt_kwargs={
                "question": "Q1",
                "answer_a": "A1",
                "answer_b": "B1",
            },
        ),
        _item(
            question_id=2,
            category="math",
            turn=2,
            prompt=multi_prompt,
            prompt_kwargs={
                "question_1": "Q2a",
                "question_2": "Q2b",
                "answer_a_1": "A2a",
                "answer_a_2": "A2b",
                "answer_b_1": "B2a",
                "answer_b_2": "B2b",
            },
        ),
    ]

    judgments, used_kwargs = infer_pairwise_judgments_by_prompt_groups(
        judge_chat_model=object(),
        items=items,
        use_tqdm=False,
        swap_answers=True,
    )

    assert judgments == ["judgment", "judgment"]
    assert used_kwargs[0]["answer_a"] == "B1"
    assert used_kwargs[0]["answer_b"] == "A1"
    assert len(captured) == 2
    assert captured[0]["input_count"] == 1
    assert captured[0]["cache_meta"]["metadata"] == [
        {
            "question_id": "1",
            "category": "writing",
            "turn": 1,
            "prompt": "default-single",
            "orientation": "reversed",
        }
    ]
    assert captured[1]["cache_meta"]["metadata"] == [
        {
            "question_id": "2",
            "category": "math",
            "turn": 2,
            "prompt": "default-multi",
            "orientation": "reversed",
        }
    ]


def test_infer_pairwise_judgments_forwards_cache(monkeypatch):
    captured: list[object] = []

    def spy_do_inference(*, cache, **kwargs):
        captured.append(cache)
        return ["judgment"] * len(kwargs["inputs"])

    monkeypatch.setattr(pairwise_module, "do_inference", spy_do_inference)

    prompt = _Prompt(
        name="default-single",
        system_prompt=None,
        user_prompt_template="{question}",
        multi_turn=False,
    )
    items = [
        _item(
            question_id=9,
            category="coding",
            turn=1,
            prompt=prompt,
            prompt_kwargs={"question": "Q", "answer_a": "A", "answer_b": "B"},
        )
    ]

    with InferenceCache("/tmp/unused", "mt-judge", mode="off") as cache:
        infer_pairwise_judgments_by_prompt_groups(
            judge_chat_model=object(),
            items=items,
            use_tqdm=False,
            swap_answers=False,
            cache=cache,
        )

    assert captured == [cache]
