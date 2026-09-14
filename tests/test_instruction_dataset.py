from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import judgearena.datasets as instruction_dataset
import judgearena.datasets.arena_hard as arena_hard
import judgearena.datasets.fluency as fluency
import judgearena.datasets.judgearena_tables as judgearena_tables
import judgearena.datasets.m_arenahard as m_arenahard
import judgearena.datasets.pairwise as pairwise_data
from judgearena.datasets.arena_hard import (
    _build_model_outputs,
    arena_hard_native_baseline,
    normalize_official_arena_hard,
)
from judgearena.tasks.registry import get_packaged_task


def test_alpaca_eval_table_download_uses_yaml_source(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        judgearena_tables, "snapshot_download", lambda **kwargs: captured.update(kwargs)
    )
    task = get_packaged_task("alpaca-eval")
    assert task is not None

    judgearena_tables.download_task_sources(task, tmp_path)

    assert captured["repo_id"] == "judge-arena/judge-arena-dataset"
    assert captured["revision"] == "004c4a992956eeefffd36b63ade470f32fd0a582"
    assert captured["allow_patterns"] == ["*alpaca-eval*"]


def test_alpaca_eval_table_loader_uses_yaml_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(
        judgearena_tables, "download_task_sources", lambda _task, _path: None
    )
    instructions_dir = tmp_path / "instructions"
    instructions_dir.mkdir()
    pd.DataFrame(
        {"instruction_index": [1, 2], "instruction": ["First", "Second"]}
    ).to_csv(instructions_dir / "alpaca-eval.csv", index=False)
    task = get_packaged_task("alpaca-eval")
    assert task is not None

    loaded = judgearena_tables.load_task_instructions(task, tmp_path)

    assert loaded["instruction_index"].tolist() == [1, 2]
    assert loaded["instruction"].tolist() == ["First", "Second"]


def test_arena_hard_native_baseline_v01_is_flat_string():
    assert arena_hard_native_baseline("arena-hard-v0.1") == "gpt-4-0314"


def test_arena_hard_native_baseline_v20_is_per_category_mapping():
    native = arena_hard_native_baseline("arena-hard-v2.0")
    assert isinstance(native, dict)
    assert native["hard_prompt"] == "o3-mini-2025-01-31"
    assert native["creative_writing"] == "gemini-2.0-flash-001"


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
    monkeypatch.setattr(
        m_arenahard, "_download_source", lambda _task, _name, _path, **_kwargs: None
    )
    monkeypatch.setattr(
        m_arenahard.pd, "read_parquet", lambda path: frames[path.parent.name].copy()
    )

    loaded = m_arenahard.load_task_instructions(task, tmp_path)

    assert loaded["instruction_index"].tolist() == ["q1-cs", "q1-uk"]
    assert loaded["instruction"].tolist() == ["Czech", "Ukrainian"]
    assert loaded["lang"].tolist() == ["cs", "uk"]


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
    monkeypatch.setattr(
        m_arenahard, "_download_source", lambda _task, _name, _path, **_kwargs: None
    )

    loaded = m_arenahard.load_task_model_outputs(task, tmp_path)

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, expected)


def test_mt_bench_native_baseline_is_flat_string():
    from judgearena.datasets.mt_bench import (
        is_mt_bench_dataset,
        mt_bench_native_baseline,
    )

    assert is_mt_bench_dataset("mt-bench") is True
    assert mt_bench_native_baseline("mt-bench") == "gpt-4"


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


def test_normalize_official_arena_hard_v20_preserves_category():
    raw_df = pd.DataFrame(
        {
            "question_id": ["q1", "q2", "q1"],
            "prompt": ["First prompt", "Second prompt", None],
            "category": ["hard_prompt", "creative_writing", None],
            "model": [None, None, "o3-mini-2025-01-31"],
            "output": [None, None, "answer text"],
        }
    )
    df_instructions, df_outputs = normalize_official_arena_hard(
        raw_df=raw_df, dataset="arena-hard-v2.0"
    )

    assert df_instructions["instruction"].tolist() == ["First prompt", "Second prompt"]
    assert df_instructions.set_index("instruction_index")["category"].to_dict() == {
        "q1": "hard_prompt",
        "q2": "creative_writing",
    }
    assert df_outputs is not None
    assert df_outputs["model"].tolist() == ["o3-mini-2025-01-31"]
    assert df_outputs["output"].tolist() == ["answer text"]


def test_build_model_outputs_extracts_upstream_messages_shape():
    """Fresh clones must keep nested answers and multiple models for one question."""
    raw_df = pd.DataFrame(
        [
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
                "uid": "q1",
                "model": "gemini-2.0-flash-001",
                "messages": [
                    {"role": "user", "content": "Prompt"},
                    {"role": "assistant", "content": "plain string answer"},
                ],
            },
            {"uid": "q3", "model": "baseline", "output": "flat output column"},
            {
                "uid": "q4",
                "model": "no-output-model",
                "messages": [{"role": "assistant", "content": {"reasoning": "..."}}],
            },
        ]
    )

    df_outputs = _build_model_outputs(raw_df)

    assert df_outputs is not None
    outputs_by_model = dict(zip(df_outputs["model"], df_outputs["output"], strict=True))
    assert outputs_by_model == {
        "o3-mini-2025-01-31": "nested answer",
        "gemini-2.0-flash-001": "plain string answer",
        "baseline": "flat output column",
    }
    assert df_outputs["instruction_index"].tolist() == ["q1", "q1", "q3"]


def test_load_instructions_uses_explicit_version_filename(monkeypatch):
    captured = {}

    def _fake_load(task, local_tables_path: Path):
        captured["dataset"] = task.task
        return pd.DataFrame(
            {
                "instruction_index": ["0", "1"],
                "instruction": ["hello", "world"],
                "category": ["hard_prompt", "creative_writing"],
            }
        )

    monkeypatch.setattr(arena_hard, "load_task_instructions", _fake_load)
    df = instruction_dataset.load_instructions(dataset="arena-hard-v2.0")

    assert captured["dataset"] == "arena-hard-v2.0"
    assert df.index.tolist() == ["0", "1"]
    assert df.loc["1", "category"] == "creative_writing"


def test_pairwise_task_data_uses_declared_adapter_outputs(monkeypatch, tmp_path):
    tables_dir = tmp_path / "tables" / "model_outputs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "arena-hard-v2.0.csv.zip"
    pd.DataFrame(
        {
            "instruction_index": [0, 0, 1, 1],
            "model": ["baseline", "candidate", "baseline", "candidate"],
            "output": ["b0", "c0", "b1", "c1"],
        }
    ).to_csv(output_path, index=False)

    adapter = SimpleNamespace(
        load_instructions=lambda task, path: pd.DataFrame(
            {"instruction_index": [0, 1], "instruction": ["q0", "q1"]}
        ),
        load_model_outputs=lambda task, path: pd.read_csv(output_path),
    )
    monkeypatch.setattr(pairwise_data, "resolve_dataset_adapter", lambda name: adapter)
    task = get_packaged_task("arena-hard-v2.0")
    assert task is not None

    task_data = pairwise_data.load_pairwise_task_data(
        task, local_tables_path=tmp_path / "tables"
    )

    loaded = task_data.model_completion("baseline")

    assert loaded is not None
    assert loaded.tolist() == ["b0", "b1"]
    assert loaded.index.tolist() == [0, 1]


def test_fluency_adapter_loads_selected_language(monkeypatch, tmp_path):
    task = get_packaged_task("fluency-french")
    assert task is not None
    monkeypatch.setattr(fluency, "download_task_sources", lambda _task, _path: None)
    root = fluency._source_local_dir(fluency._source(task), tmp_path)
    (root / "French").mkdir(parents=True)
    pd.DataFrame({"sentence": ["Le chat", "La maison"]}).to_parquet(
        root / "French" / "data.parquet"
    )

    loaded = fluency.load_task_instructions(task, tmp_path)

    assert loaded["instruction"].tolist() == ["Le chat", "La maison"]
    assert loaded["instruction_index"].tolist() == ["french-0", "french-1"]
