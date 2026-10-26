from dataclasses import replace

import pandas as pd
import pytest

import judgearena.datasets as instruction_dataset
import judgearena.datasets.arena_hard as arena_hard
import judgearena.datasets.data as pairwise_data
import judgearena.datasets.fluency as fluency
import judgearena.datasets.judgearena_tables as judgearena_tables
import judgearena.datasets.m_arenahard as m_arenahard
from judgearena.datasets.arena_hard import (
    normalize_official_arena_hard,
)
from judgearena.tasks.registry import get_packaged_task


@pytest.fixture
def battle_rows():
    return pd.DataFrame(
        {
            "instruction": ["one", "two", "three"],
            "question_id": "same",
            "model_a": "a",
            "model_b": "b",
            "completion_a": ["a1", "a2", "a3"],
            "completion_b": ["b1", "b2", "b3"],
            "pref": [0.0, 0.5, 1.0],
            "source": "human",
        }
    )


def test_table_adapter_reuses_and_normalizes_alpaca_tables(monkeypatch, tmp_path):
    downloads = []
    monkeypatch.setattr(
        judgearena_tables,
        "snapshot_download",
        lambda **kwargs: downloads.append(kwargs),
    )
    instructions_dir = tmp_path / "instructions"
    outputs_dir = tmp_path / "model_outputs"
    instructions_dir.mkdir()
    outputs_dir.mkdir()
    pd.DataFrame(
        {
            "instruction_index": ["slug-z", "slug-a"],
            "instruction": ["First", "Second"],
            "domain": ["helpful_base", "koala"],
        }
    ).to_csv(instructions_dir / "alpaca-eval.csv", index=False)
    pd.DataFrame(
        {
            "instruction_index": ["slug-z", "slug-a", "slug-z"],
            "model": ["gpt4_1106_preview", "gpt4_1106_preview", "archived-model"],
            "output": ["base-0", "base-1", "archived-0"],
        }
    ).to_csv(outputs_dir / "alpaca-eval.csv.zip", index=False)
    task = get_packaged_task("alpaca-eval")

    data = instruction_dataset.load_task_data(task, local_tables_path=tmp_path)
    instructions = data.instructions.reset_index()
    outputs = data.model_outputs

    assert len(downloads) == 1
    assert instructions.to_dict(orient="records") == [
        {
            "instruction_index": 0,
            "instruction": "First",
            "category": "helpful_base",
        },
        {"instruction_index": 1, "instruction": "Second", "category": "koala"},
    ]
    assert outputs.to_dict(orient="records") == [
        {"instruction_index": 0, "model": "gpt4_1106_preview", "output": "base-0"},
        {"instruction_index": 1, "model": "gpt4_1106_preview", "output": "base-1"},
        {"instruction_index": 0, "model": "archived-model", "output": "archived-0"},
    ]


@pytest.mark.parametrize("observed", [False, True])
def test_table_adapter_loads_mapped_battles_with_repeated_question_ids(
    monkeypatch, tmp_path, observed, battle_rows
):
    monkeypatch.setattr(judgearena_tables, "snapshot_download", lambda **kwargs: None)
    task = get_packaged_task("alpaca-eval")
    mapping = {
        "prompt": "instruction",
        "ours": "model_a",
        "theirs": "model_b",
        "answer": "completion_a",
        "reference": "completion_b",
        "vote": "pref",
    }
    spec = task.spec.model_dump()
    spec["dataset"].update(
        format="battles",
        fields=None,
        battle_columns=mapping,
        outcome_preferences={"left": 0.0, "right": 1.0},
    )
    spec["dataset"]["sources"]["tables"]["allow_patterns"] = ["battles/custom.csv"]
    task = replace(task, spec=type(task.spec).model_validate(spec))
    (tmp_path / "battles").mkdir()
    rows = battle_rows.iloc[:2].assign(source=["human", "llm-judge"], pref=None)
    if observed:
        rows = rows.drop(columns="pref").assign(winner=["left", "right"])
    rows.rename(columns={v: k for k, v in mapping.items()}).to_csv(
        tmp_path / "battles/custom.csv",
        index=False,
    )

    data = instruction_dataset.load_task_data(task, local_tables_path=tmp_path)

    assert data.instructions.index.tolist() == [0, 1]
    assert data.battles["question_id"].tolist() == ["same", "same"]
    assert data.model_completion("a").tolist() == ["a1", "a2"]
    assert data.model_completion("b").tolist() == ["b1", "b2"]
    if observed:
        assert data.human_battles()["pref"].tolist() == [0.0]
    else:
        with pytest.raises(ValueError, match="human preference labels"):
            data.human_battles()

    selected = data.select({"source": ["human"]})
    assert selected.instructions.index.tolist() == [0]
    assert selected.model_completion("a").to_dict() == {0: "a1"}
    assert selected.battles["source"].tolist() == ["human"]


def test_m_arena_hard_adapter_filters_selected_language_group(monkeypatch, tmp_path):
    task = get_packaged_task("m-arena-hard-v2.0-EU")
    assert task is not None
    source = task.spec.dataset.sources["examples"]
    source_root = tmp_path / "_sources" / source.repo_id.replace("/", "--")
    for language in ("cs", "uk", "ar"):
        language_dir = source_root / language
        language_dir.mkdir(parents=True)
        (language_dir / "test.parquet").touch()

    frames = {
        "cs": pd.DataFrame({"question_id": ["q1"], "prompt": ["Czech"]}),
        "uk": pd.DataFrame({"question_id": ["q1"], "prompt": ["Ukrainian"]}),
        "ar": pd.DataFrame({"question_id": ["q1"], "prompt": ["Arabic"]}),
    }
    downloads = []
    monkeypatch.setattr(
        m_arenahard, "snapshot_download", lambda **kw: downloads.append(kw)
    )
    monkeypatch.setattr(
        m_arenahard.pd,
        "read_parquet",
        lambda path: frames[path.parent.name].copy(),
    )

    loaded = m_arenahard.MArenaHardAdapter(task, tmp_path).load_instructions()

    assert loaded["instruction_index"].tolist() == ["q1-cs", "q1-uk"]
    assert loaded["instruction"].tolist() == ["Czech", "Ukrainian"]
    assert loaded["lang"].tolist() == ["cs", "uk"]
    assert downloads[0]["local_dir"] == source_root


def test_m_arena_hard_adapter_loads_invocation_specific_outputs(monkeypatch, tmp_path):
    task = get_packaged_task("m-arena-hard-v0.1-uk")
    assert task is not None
    output_path = tmp_path / "model_outputs" / f"{task.task}.csv.zip"
    output_path.parent.mkdir()
    expected = pd.DataFrame(
        {
            "instruction_index": ["q1-uk"],
            "model": ["CohereLabs/aya-expanse-8b"],
            "output": ["answer"],
        }
    )
    expected.to_csv(output_path, index=False)
    downloads = []
    monkeypatch.setattr(
        m_arenahard, "snapshot_download", lambda **kw: downloads.append(kw)
    )

    loaded = m_arenahard.MArenaHardAdapter(task, tmp_path).load_model_outputs()

    assert downloads[0]["local_dir"] == tmp_path
    assert downloads[0]["allow_patterns"] == [f"model_outputs/{task.task}.csv.zip"]
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, expected)


def test_normalize_official_arena_hard_v01_drops_no_category():
    raw_df = pd.DataFrame(
        {
            "question_id": ["q1", "q2"],
            "prompt": ["First prompt", "Second prompt"],
            "model": ["baseline", "baseline"],
            "output": ["a1", "a2"],
        }
    )
    df_instructions, df_outputs = normalize_official_arena_hard(
        raw_df=raw_df, dataset="arena-hard-v0.1"
    )

    assert df_instructions.columns.tolist() == ["instruction_index", "instruction"]
    assert df_instructions["instruction_index"].tolist() == ["q1", "q2"]
    assert df_instructions["instruction"].tolist() == ["First prompt", "Second prompt"]
    assert df_outputs is not None
    assert set(df_outputs.columns) == {"instruction_index", "model", "output"}


def test_arena_hard_normalizes_mixed_questions_and_answers():
    raw_df = pd.DataFrame(
        [
            {"uid": "q1", "prompt": "real prompt", "category": "hard_prompt"},
            {
                "uid": "q1",
                "model": "o3-mini-2025-01-31",
                "messages": [
                    {"role": "user", "content": "Prompt"},
                    {
                        "role": "assistant",
                        "content": {"answer": "nested answer", "reasoning": "..."},
                    },
                ],
            },
            {
                "uid": "q2",
                "model": "gemini-2.0-flash-001",
                "messages": [
                    {"role": "user", "content": "Prompt"},
                    {"role": "assistant", "content": "plain string answer"},
                ],
            },
            {
                "uid": "q3",
                "model": "baseline",
                "output": "flat output column",
            },
            {"uid": "q1", "model": "baseline", "output": "second model q1"},
            {"uid": "q5", "model": "missing", "output": None, "messages": None},
            {
                "uid": "q4",
                "model": "no-output-model",
                "messages": [{"role": "assistant", "content": {"reasoning": "..."}}],
            },
        ]
    )

    instructions, df_outputs = normalize_official_arena_hard(raw_df, "arena-hard-v2.0")
    assert instructions[
        ["instruction_index", "instruction", "category"]
    ].values.tolist() == [["q1", "real prompt", "hard_prompt"]]

    assert df_outputs.set_index(["instruction_index", "model"])["output"].to_dict() == {
        ("q1", "o3-mini-2025-01-31"): "nested answer",
        ("q2", "gemini-2.0-flash-001"): "plain string answer",
        ("q3", "baseline"): "flat output column",
        ("q1", "baseline"): "second model q1",
    }


def test_pairwise_task_data_loads_cached_tables_and_selected_outputs(tmp_path):
    task = get_packaged_task("arena-hard-v2.0")
    instruction_path, output_path = arena_hard.ArenaHardAdapter(
        task, tmp_path / "tables"
    )._table_paths()
    instruction_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "instruction_index": [0, 1],
            "instruction": ["q0", "q1"],
            "category": ["hard_prompt", "creative_writing"],
        }
    ).to_csv(instruction_path, index=False)
    pd.DataFrame(
        {
            "instruction_index": [0, 1, 1],
            "model": ["baseline", "baseline", "category-baseline"],
            "output": ["b0", "b1", "answer"],
        }
    ).to_csv(output_path, index=False)

    task_data = instruction_dataset.load_task_data(
        task, local_tables_path=tmp_path / "tables"
    )

    loaded = task_data.model_completion("baseline")

    assert loaded is not None
    assert loaded.tolist() == ["b0", "b1"]
    assert loaded.index.tolist() == [0, 1]
    assert task_data.instructions["category"].to_dict() == {
        0: "hard_prompt",
        1: "creative_writing",
    }

    loaded = task_data.model_completion(
        "category-baseline", instruction_ids=pd.Index([1], name="instruction_index")
    )

    assert loaded is not None
    assert loaded.to_dict() == {1: "answer"}
    missing = task_data.model_completion(
        "category-baseline", instruction_ids=pd.Index([0], name="instruction_index")
    )
    assert missing.isna().all()


def test_fluency_variants_cover_all_supported_languages():
    task = get_packaged_task("fluency")
    assert task is not None
    expected = {
        fluency.fluency_language_slug(language)
        for language in fluency.FLUENCY_LANGUAGES
    }
    assert set(task.spec.variants.values) == expected


def test_fluency_adapter_loads_selected_language(monkeypatch, tmp_path):
    task = get_packaged_task("fluency-french")
    assert task is not None
    monkeypatch.setattr(fluency.FluencyAdapter, "download", lambda self: None)
    root = fluency._source_local_dir(
        fluency.FluencyAdapter(task, tmp_path)._source(), tmp_path
    )
    (root / "French").mkdir(parents=True)
    pd.DataFrame({"sentence": ["Le chat", "La maison"]}).to_parquet(
        root / "French" / "data.parquet"
    )

    loaded = fluency.FluencyAdapter(task, tmp_path).load_instructions()

    assert loaded["instruction"].tolist() == ["Le chat", "La maison"]
    assert loaded["instruction_index"].tolist() == ["french-0", "french-1"]


@pytest.mark.parametrize("bad_pref", [-0.1, 1.1, float("inf")])
def test_preference_data_rejects_invalid_preferences(bad_pref, battle_rows):
    with pytest.raises(ValueError, match="preferences"):
        pairwise_data.PreferenceData.from_battles(battle_rows.assign(pref=bad_pref))


def test_completion_data_rejects_outputs_for_unknown_instructions():
    from judgearena.datasets.data import CompletionData

    with pytest.raises(ValueError, match="unknown instruction"):
        CompletionData.from_frames(
            pd.DataFrame({"id": [1], "prompt": ["q"]}),
            pd.DataFrame({"id": [2], "model": ["a"], "output": ["answer"]}),
            instruction_columns={"id": "instruction_index", "prompt": "instruction"},
            output_columns={"id": "instruction_index"},
        )


def test_preference_data_rejects_unmapped_or_conflicting_outcomes(battle_rows):
    raw = battle_rows.assign(winner="right")
    with pytest.raises(ValueError, match="outcome_preferences"):
        pairwise_data.PreferenceData.from_battles(raw.drop(columns="pref"))
    with pytest.raises(ValueError, match="conflict"):
        pairwise_data.PreferenceData.from_battles(
            raw, outcome_preferences={"right": 1.0}
        )
