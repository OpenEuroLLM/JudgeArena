import pandas as pd

import judgearena.browse_cache as browser
from judgearena.browse_cache import cell_row_count, iter_cache_cells, load_cache_cell
from judgearena.cache_sqlite import (
    COMPLETION_DB_NAME,
    JUDGEMENT_DB_NAME,
    CompletionCache,
    JudgementCache,
    cache_folder,
    write_descriptor,
)


def test_browser_discovers_and_joins_content_addressed_cells(tmp_path, monkeypatch):
    descriptor = {
        "schema_version": "judgearena-inference-cache/v1",
        "provider": "Dummy",
        "model": "org/model",
    }
    folder = cache_folder(
        tmp_path,
        "completions",
        "arena-hard",
        "Dummy/org/model",
        descriptor,
    )
    write_descriptor(folder, descriptor)
    rows = pd.DataFrame(
        [
            {
                "input_text": "prompt",
                "completion": "answer",
                "benchmark": "arena-hard",
                "instruction_id": "7",
                "model": "Dummy/org/model",
            }
        ]
    )
    with CompletionCache(folder / COMPLETION_DB_NAME) as cache:
        cache.save(rows, pushed_by="test")
    judgement_folder = cache_folder(
        tmp_path,
        "judgements",
        "arena-hard",
        "Dummy/judge",
        {**descriptor, "model": "judge"},
    )
    write_descriptor(judgement_folder, {**descriptor, "model": "judge"})
    with JudgementCache(judgement_folder / JUDGEMENT_DB_NAME) as cache:
        cache.save(
            pd.DataFrame(
                [
                    {
                        "judge_input": "judge prompt",
                        "judge_completion": "score_A: 1\nscore_B: 9",
                        "benchmark": "arena-hard",
                        "instruction_id": "7",
                        "model_a": "baseline",
                        "model_b": "Dummy/org/model",
                        "judge": "Dummy/judge",
                        "orientation": "reversed",
                    }
                ]
            ),
            pushed_by="test",
        )

    cells = iter_cache_cells(tmp_path)

    assert len(cells) == 2
    assert cell_row_count(cells[0]) == 1
    assert load_cache_cell(cells[0], instruction_id="7")["completion"].tolist() == [
        "answer"
    ]
    context = pd.DataFrame(
        {"instruction": ["question"], "language": ["en"]},
        index=["7"],
    )
    monkeypatch.setattr(browser, "load_context", lambda _task: context)
    joined = browser.load_subset(
        store_root=tmp_path,
        task="arena-hard",
        model="Dummy/org/model",
        judge="Dummy/judge",
        languages=["en"],
    )

    assert joined["completion_b"].tolist() == ["answer"]
    assert joined["instruction"].tolist() == ["question"]
    assert "score_A: 1" in browser.render_row(joined.iloc[0])
