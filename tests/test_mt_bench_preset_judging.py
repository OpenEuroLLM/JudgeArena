from __future__ import annotations

from functools import partial

import pandas as pd
import pytest
from langchain_core.language_models.fake import FakeListLLM

from judgearena.benchmarks.pairwise.comparisons import conversation_comparisons
from judgearena.cache.inference import JudgementInferenceCache
from judgearena.judging import execute_pairwise
from judgearena.models import do_inference, prepare_model
from judgearena.prompts.mt_bench import (
    build_mt_bench_preset_prompts,
    render_mt_bench_requests,
)
from judgearena.prompts.registry import (
    resolve_judge_prompt,
)

REFERENCE_CATEGORIES = ("math", "reasoning", "coding", "arena-hard-200")


def test_grouped_judging_reuses_raw_cache_and_restores_item_order(
    tmp_path, monkeypatch
):
    from judgearena.prompts.parsing import JUDGE_PARSERS, PairScore

    monkeypatch.setitem(JUDGE_PARSERS, "custom", PairScore(temperature=0.5))
    system_file = tmp_path / "system.txt"
    system_file.write_text("Custom system")
    user_file = tmp_path / "prompt.txt"
    user_file.write_text(
        "Custom rubric: {question} {answer_a} {answer_b} {ref_answer_1} Score each {completion_label}."
    )
    prompt = resolve_judge_prompt(
        system_file=system_file, user_file=user_file, parser="custom"
    )
    prompts = build_mt_bench_preset_prompts({False: prompt})
    system_file.unlink()
    user_file.unlink()
    comparisons = conversation_comparisons(
        pd.DataFrame(
            {
                "category": ["math", "writing", "math"],
                "reference_turn_1": "R1",
                "model_a": "a",
                "model_b": "b",
                **{
                    f"messages_{slot}": [
                        [
                            {"role": "user", "content": f"Question {i}"},
                            {"role": "assistant", "content": slot},
                        ]
                        for i in [1, 2, 3]
                    ]
                    for slot in ["a", "b"]
                },
            },
            index=[1, 2, 3],
        ),
        [1],
    )
    render = partial(
        render_mt_bench_requests,
        select_prompt=lambda category, multi: prompts[
            multi, category in REFERENCE_CATEGORIES
        ],
    )
    batches = list(render(comparisons))
    assert [[r.id for r in batch] for batch in batches] == [
        ["1:turn-1", "3:turn-1"],
        ["2:turn-1"],
    ]
    requests = {r.id: r for batch in batches for r in batch}
    raw = ["score_A: 10\nscore_B: 0", "no verdict", "score_A: 8\nscore_B: 2"]
    cache = JudgementInferenceCache(tmp_path, "mt-bench")
    model = prepare_model("Dummy/judge", cache=cache)
    model.factory = lambda: FakeListLLM(responses=raw)
    metadata = [
        {
            "instruction_id": identity,
            "model_a": "a",
            "model_b": "b",
            "orientation": "direct",
        }
        for identity in comparisons.index
    ]
    # Seed the pre-executor inference format, in item order rather than prompt groups.
    do_inference(
        model,
        [requests[i].input for i in comparisons.index],
        stage="judging",
        cache_row_metadata=metadata,
    )
    cached_model = prepare_model("Dummy/judge", cache=cache)

    def unavailable():
        raise AssertionError("Existing MT-Bench judgments should be cache hits")

    cached_model.factory = unavailable
    results, _ = execute_pairwise(cached_model, comparisons, render)
    assert [result.response.text for result in results] == raw
    assert [
        None if result.parsed is None else result.parsed.scores for result in results
    ] == [{"A": 10, "B": 0}, None, {"A": 8, "B": 2}]
    assert [result.request.cache_metadata for result in results] == metadata
    inputs = [result.request.input.to_string() for result in results]
    assert all("Custom rubric:" in text for text in inputs)
    assert "R1" in inputs[0]
    assert results[2].parsed.preference == pytest.approx(0.0474258732)


@pytest.mark.parametrize("multi_turn", [False, True])
def test_mt_builder_rejects_generic_template_fields_early(multi_turn):
    prompt = resolve_judge_prompt(preset="arena-hard")
    with pytest.raises(ValueError, match="MT-Bench.*completion_A"):
        build_mt_bench_preset_prompts({multi_turn: prompt})
