from __future__ import annotations

import math

import judgearena.evaluate as evaluate_module
from judgearena.evaluate import annotate_battles, judge_and_parse_prefs
from judgearena.inference_cache import InferenceCache
from judgearena.models import make_model


class FakeJudge:
    def __init__(self, response: str = "score A: 0 score B: 10"):
        self.response = response

    def batch(self, *, inputs, **_kwargs):
        return [self.response] * len(inputs)


def test_annotate_battles_forwards_cache_and_metadata(monkeypatch):
    captured: list[dict] = []

    def spy_do_inference(*, cache, cache_meta, **kwargs):
        captured.append({"cache": cache, "cache_meta": cache_meta})
        return ["score A: 0 score B: 10"]

    monkeypatch.setattr(evaluate_module, "do_inference", spy_do_inference)

    row_metadata = [{"battle_id": "b-1"}]
    with InferenceCache("/tmp/unused", "judge", mode="off") as cache:
        annotate_battles(
            judge_chat_model=FakeJudge(),
            instructions=["Question"],
            completions_A=["A"],
            completions_B=["B"],
            cache=cache,
            row_metadata=row_metadata,
        )

    assert captured[0]["cache"] is cache
    assert captured[0]["cache_meta"] == {"metadata": row_metadata}


def test_judge_and_parse_prefs_adds_orientation_and_forwards_both(monkeypatch):
    captured: list[dict] = []

    def spy_do_inference(*, cache_meta, **kwargs):
        captured.append(cache_meta)
        return ["score A: 0 score B: 10"] * len(kwargs["inputs"])

    monkeypatch.setattr(evaluate_module, "do_inference", spy_do_inference)

    base_metadata = [{"question_id": "q-42"}]

    _, annotations_reversed, prefs = judge_and_parse_prefs(
        judge_chat_model=FakeJudge(),
        instructions=["Q1", "Q2"],
        completions_A=["A1", "A2"],
        completions_B=["B1", "B2"],
        swap_mode="both",
        row_metadata=base_metadata * 2,
    )

    assert annotations_reversed is not None
    assert len(captured) == 2
    assert captured[0]["metadata"] == [
        {"question_id": "q-42", "orientation": "direct"},
        {"question_id": "q-42", "orientation": "direct"},
    ]
    assert captured[1]["metadata"] == [
        {"question_id": "q-42", "orientation": "reversed"},
        {"question_id": "q-42", "orientation": "reversed"},
    ]
    assert len(prefs) == 4


def test_judge_and_parse_prefs_default_without_cache_unchanged():
    judge = make_model("Dummy/score A: 0 score B: 10")
    _, annotations_reversed, prefs = judge_and_parse_prefs(
        judge_chat_model=judge,
        instructions=["Q"],
        completions_A=["A"],
        completions_B=["B"],
        swap_mode="fixed",
    )

    assert annotations_reversed is None
    assert len(prefs) == 1
    assert not math.isnan(float(prefs.iloc[0]))


def test_annotate_battles_without_metadata_omits_cache_meta(monkeypatch):
    captured: list[dict] = []

    def spy_do_inference(*, cache_meta=None, **kwargs):
        captured.append({"cache_meta": cache_meta})
        return ["score A: 0 score B: 10"]

    monkeypatch.setattr(evaluate_module, "do_inference", spy_do_inference)

    annotate_battles(
        judge_chat_model=FakeJudge(),
        instructions=["Question"],
        completions_A=["A"],
        completions_B=["B"],
    )

    assert captured[0]["cache_meta"] is None
