from dataclasses import replace

import pandas as pd
import pytest

import judgearena.datasets as instruction_dataset
import judgearena.datasets.arena_hard as arena_hard
import judgearena.datasets.data as pairwise_data
import judgearena.datasets.fluency as fluency
import judgearena.datasets.judgearena_tables as judgearena_tables
import judgearena.datasets.m_arenahard as m_arenahard
from judgearena.datasets.arena_battles import normalize_battles
from judgearena.datasets.arena_hard import (
    _build_instructions,
    _build_model_outputs,
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
            "reference_pref": [0.0, 0.5, 1.0],
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
        {"instruction_index": 0, "model": "gpt4_1106_preview", "completion": "base-0"},
        {"instruction_index": 1, "model": "gpt4_1106_preview", "completion": "base-1"},
        {"instruction_index": 0, "model": "archived-model", "completion": "archived-0"},
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
        "vote": "reference_pref",
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
    rows = battle_rows.iloc[:2].assign(
        source=["human", "llm-judge"], reference_pref=None
    )
    if observed:
        rows = rows.drop(columns="reference_pref").assign(winner=["left", "right"])
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
        assert data.battles["reference_pref"].tolist() == [0.0, 1.0]
    else:
        assert data.battles["reference_pref"].isna().all()

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
    pd.testing.assert_frame_equal(
        loaded, expected.rename(columns={"output": "completion"})
    )


def test_arena_hard_v01_normalization_without_category():
    raw_df = pd.DataFrame(
        {
            "question_id": ["q1", "q2"],
            "prompt": ["First prompt", "Second prompt"],
            "model": ["baseline", "baseline"],
            "output": ["a1", "a2"],
        }
    )
    df_instructions = _build_instructions(raw_df)
    df_outputs = _build_model_outputs(raw_df)

    assert df_instructions.columns.tolist() == ["instruction_index", "instruction"]
    assert df_instructions["instruction_index"].tolist() == ["q1", "q2"]
    assert df_instructions["instruction"].tolist() == ["First prompt", "Second prompt"]
    assert df_outputs is not None
    assert set(df_outputs.columns) == {"instruction_index", "model", "completion"}


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

    instructions = _build_instructions(raw_df)
    df_outputs = _build_model_outputs(raw_df)
    assert instructions[
        ["instruction_index", "instruction", "category"]
    ].values.tolist() == [["q1", "real prompt", "hard_prompt"]]

    assert df_outputs.set_index(["instruction_index", "model"])[
        "completion"
    ].to_dict() == {
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
        pairwise_data.PreferenceData.from_battles(
            battle_rows.assign(reference_pref=bad_pref)
        )


@pytest.mark.parametrize("unknown", [2, "1"])
def test_completion_data_rejects_outputs_for_unknown_instructions(unknown):
    with pytest.raises(ValueError, match="unknown instruction") as error:
        pairwise_data.CompletionData(
            pd.DataFrame({"instruction": ["q"]}, index=[1]),
            pd.DataFrame(
                {
                    "instruction_index": [unknown],
                    "model": ["a"],
                    "completion": ["answer"],
                }
            ),
        )

    assert f"{unknown!r} ({type(unknown).__name__})" in str(error.value)
    assert "1 (int)" in str(error.value)


@pytest.mark.parametrize("labeled", [False, True])
def test_adapter_frames_preserve_metadata_and_alignment(battle_rows, labeled):
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "a1"},
    ]
    rows = battle_rows.iloc[:1].assign(
        instruction_index=8,
        checkpoint_a="saved",
        messages_a=[messages],
        context=[{"rubric": ["grammar"]}],
    )
    rows = pd.concat(
        [rows, rows.assign(model_b="c", completion_b="c1", reference_pref=1.0)],
        ignore_index=True,
    )
    rows["battle_id"] = ["ab", "ac"]
    rows["tstamp"] = [1, 2]
    if not labeled:
        rows = rows.drop(columns=["reference_pref", "source"])
    data = pairwise_data.PreferenceData.from_battles(rows)
    assert "reference_pref" not in data.instructions
    if labeled:
        assert data.battles["reference_pref"].tolist() == [0.0, 1.0]
    loaded = data.model_output_rows("a").iloc[0]
    assert "tstamp" not in data.instructions
    assert data.battles["tstamp"].tolist() == [1, 2]
    assert loaded["completion"] == "a1"
    assert loaded["messages"] == messages
    assert loaded["checkpoint"] == "saved"
    assert "messages_a" not in data.instructions
    assert data.model_completion("b").to_dict() == {8: "b1"}
    assert data.instructions["question_id"].tolist() == ["same"]
    assert data.model_completion("c").to_dict() == {8: "c1"}
    assert data.battles["model_b"].tolist() == ["b", "c"]
    assert len(data.model_outputs) == 4
    assert data.battles["battle_id"].tolist() == ["ab", "ac"]
    assert "battle_id" not in data.instructions
    conflicting = rows.copy()
    conflicting.at[1, "instruction"] = "different"
    with pytest.raises(ValueError, match="conflicting instruction"):
        pairwise_data.PreferenceData.from_battles(conflicting)
    conflicting = rows.copy()
    conflicting.at[1, "context"] = {"rubric": ["logic"]}
    assert (
        "context"
        not in pairwise_data.PreferenceData.from_battles(conflicting).instructions
    )


def test_preference_mapping_rejects_conflicting_source_labels(battle_rows):
    with pytest.raises(ValueError, match="conflict"):
        pairwise_data.normalize_preferences(
            battle_rows.assign(winner="right"), {"right": 1.0}
        )


@pytest.mark.parametrize("second_answer", ["first", "different"])
def test_answer_lookup_rejects_only_requested_ambiguity(second_answer):
    data = pairwise_data.CompletionData(
        pd.DataFrame({"instruction": ["Q1", "Q2", "Q3"]}, index=[1, 2, 3]),
        pd.DataFrame(
            {
                "instruction_index": [1, 1, 2, 3],
                "model": "a",
                "completion": ["first", second_answer, None, ""],
            }
        ),
    )
    selected = data.model_output_rows("a", instruction_ids=pd.Index([3, 2, 3]))
    assert selected["completion"].iloc[0] == selected["completion"].iloc[2] == ""
    assert pd.isna(selected["completion"].iloc[1])
    if second_answer == "first":
        assert data.model_completion("a", instruction_ids=pd.Index([1])).tolist() == [
            "first"
        ]
    else:
        with pytest.raises(ValueError, match="multiple saved answers"):
            data.model_output_rows("a", instruction_ids=pd.Index([1]))


@pytest.mark.parametrize("mismatched_prompt", [True, False])
def test_arena_adapter_filters_invalid_pairs_without_exposing_system_identity(
    battle_rows, mismatched_prompt
):
    history = [
        {"role": "system", "content": "Answer in French"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "a1"},
    ]
    frame = battle_rows.copy()
    frame["conversation_a"] = [history, history[1:], history]
    frame["conversation_b"] = [
        [history[1], {"role": "assistant", "content": "b1"}],
        history[1:],
        history[:1] + [{"role": "user", "content": "different"}, history[-1]]
        if mismatched_prompt
        else [],
    ]
    data = normalize_battles(frame, "test-arena")
    from judgearena.benchmarks.pairwise.judging import comparison_text

    assert data.battles["instruction"].tolist() == ["one", "one"]
    rendered = comparison_text(data.battles, strip_thinking=False)
    assert rendered["completion_a"].tolist() == ["a1", "a1"]
    assert rendered["completion_b"].tolist() == ["b1", "a1"]
    assert data.battles["conversation_a"].iloc[0] == history


def test_supplied_history_validation_and_system_override():
    from judgearena.generate import generate_messages

    history = [{"role": "system", "content": "Old"}, {"role": "user", "content": "Q"}]
    data = pairwise_data.CompletionData(
        pd.DataFrame({"instruction": ["Q"], "messages": [history]})
    )
    result = generate_messages(
        data.instructions["messages"], "Dummy/answer", system_prompt="New"
    )
    assert result.iloc[0]["messages"] == [
        {"role": "system", "content": "New"},
        history[-1],
        {"role": "assistant", "content": "answer"},
    ]
    with pytest.raises(ValueError, match="end.*user"):
        pairwise_data.CompletionData(
            pd.DataFrame(
                {"instruction": ["Q"], "messages": [result.iloc[0]["messages"]]}
            )
        )
    with pytest.raises(ValueError, match="expected"):
        generate_messages(pd.Series([[history[-1], history[-1]]]), "Dummy/answer")


def test_comparison_ids_survive_selection_with_shared_instructions(battle_rows):
    rows = battle_rows.iloc[:2].assign(instruction_index="q", instruction="same")
    data = pairwise_data.PreferenceData.from_battles(rows)
    assert data.battles.battle_id.tolist() == ["0", "1"]
    assert data.select({"reference_pref": [0.5]}).battles.battle_id.tolist() == ["1"]


@pytest.mark.parametrize("task_name", ["mt-bench-101", "wildbench"])
def test_fixed_format_adapters_reject_unused_field_mappings(
    tmp_path, monkeypatch, task_name
):
    from judgearena.datasets.registry import resolve_dataset_adapter

    task = get_packaged_task(task_name)
    spec = task.spec.model_dump()
    spec["dataset"]["fields"] = {"id": "qid", "instruction": "prompt"}
    task = replace(task, spec=type(task.spec).model_validate(spec))
    adapter = resolve_dataset_adapter(task, tmp_path)
    method = "load_instructions" if task_name == "mt-bench-101" else "_load_examples"
    monkeypatch.setattr(
        adapter, method, lambda: pytest.fail("Reject before reading source data")
    )
    with pytest.raises(ValueError, match="dataset.fields.*custom adapter"):
        adapter.load_data()


def test_arena_loading_keeps_only_complete_first_turns(monkeypatch):
    import judgearena.arenas_utils as source

    pair = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    histories = [
        pair,
        [{"role": "system", "content": "S"}, *pair],
        [*pair, {"role": "user", "content": "unfinished"}],
        [{"role": "user", "content": 123}, pair[-1]],
    ]
    rows = pd.DataFrame({"conversation_a": histories, "conversation_b": histories})
    rows = rows.assign(
        id=range(4),
        timestamp=0,
        model_a_name="a",
        model_b_name="b",
        chosen_model_name="a",
        both_equal=False,
    )
    monkeypatch.setattr(source, "_download_arena_dataset", lambda **kwargs: "/unused")
    monkeypatch.setattr(source.pd, "read_parquet", lambda path: rows)
    monkeypatch.setattr(source, "detect_language", lambda text: "en")
    loaded = source._load_arena_dataframe("ComparIA", dataset_sources={})
    assert loaded.question_id.tolist() == [0, 1, 3]
    normalized = normalize_battles(loaded, "ComparIA")
    assert normalized.battles.question_id.tolist() == [0, 1]
