import pandas as pd

import judgearena.instruction_dataset.mt_bench as mt_bench
import judgearena.mt_bench.mt_bench_utils as mt_bench_utils
import judgearena.utils as utils
from judgearena.generate_and_evaluate import CliArgs


def test_download_mt_bench_skips_question_download_if_cached(tmp_path, monkeypatch):
    question_path = tmp_path / "data" / "mt_bench" / "question.jsonl"
    question_path.parent.mkdir(parents=True, exist_ok=True)
    question_path.write_text('{"question_id": 1, "turns": ["Q1"]}\n')

    reference_path = tmp_path / "reference_answer" / "gpt-4.jsonl"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text('{"question_id": 1, "choices": [{"turns": ["A1"]}]}\n')

    calls = {"snapshot_download": 0}

    def _snapshot_download_stub(**_kwargs):
        calls["snapshot_download"] += 1

    monkeypatch.setattr(mt_bench, "snapshot_download", _snapshot_download_stub)
    monkeypatch.setattr(
        mt_bench,
        "_download_gpt4_references",
        lambda _local_dir: reference_path,
    )

    downloaded_question_path, downloaded_reference_path = mt_bench.download_mt_bench(
        local_dir=tmp_path
    )

    assert downloaded_question_path == question_path
    assert downloaded_reference_path == reference_path
    assert calls["snapshot_download"] == 0


def test_download_all_includes_mt_bench(tmp_path, monkeypatch):
    hf_datasets = []
    arena_hard_datasets = []
    calls = {"contexts": 0, "mt_bench": 0}

    monkeypatch.setattr(utils, "data_root", tmp_path)
    monkeypatch.setattr(
        utils,
        "download_hf",
        lambda name, local_path: hf_datasets.append((name, local_path)),
    )
    monkeypatch.setattr(
        utils,
        "download_arena_hard",
        lambda dataset, local_tables_path: arena_hard_datasets.append(
            (dataset, local_tables_path)
        ),
    )

    def _contexts_snapshot_stub(**_kwargs):
        calls["contexts"] += 1

    monkeypatch.setattr(utils, "snapshot_download", _contexts_snapshot_stub)
    monkeypatch.setattr(
        mt_bench,
        "download_mt_bench",
        lambda: calls.__setitem__("mt_bench", calls["mt_bench"] + 1),
    )

    utils.download_all()

    tables_dir = tmp_path / "tables"
    assert [name for name, _ in hf_datasets] == [
        "alpaca-eval",
        "m-arena-hard-v0.1",
        "m-arena-hard-v2.0",
    ]
    assert arena_hard_datasets == [
        ("arena-hard-v0.1", tables_dir),
        ("arena-hard-v2.0", tables_dir),
    ]
    assert calls["contexts"] == 1
    assert calls["mt_bench"] == 1


def test_load_mt_bench_model_answers_reads_cached_baseline_file(tmp_path):
    answer_path = tmp_path / "data" / "mt_bench" / "model_answer" / "gpt-4.jsonl"
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(
        '{"question_id": 2, "choices": [{"turns": ["A2", "B2"]}]}\n'
        '{"question_id": 1, "choices": [{"turns": ["A1"]}]}\n'
    )

    df_answers = mt_bench.load_mt_bench_model_answers("gpt-4", local_dir=tmp_path)

    assert df_answers.to_dict(orient="records") == [
        {
            "instruction_index": 1,
            "completion_turn_1": "A1",
            "completion_turn_2": "",
        },
        {
            "instruction_index": 2,
            "completion_turn_1": "A2",
            "completion_turn_2": "B2",
        },
    ]


def test_generate_mt_bench_completions_uses_pregenerated_baseline(monkeypatch):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1", "Q2"], "turn_2": ["Q1b", "Q2b"]},
        index=pd.Index([1, 2], name="instruction_index"),
    )
    generated_models = []

    monkeypatch.setattr(
        mt_bench_utils, "cache_function_dataframe", lambda fun, **_kwargs: fun()
    )

    def fake_generate_multiturn(**kwargs):
        generated_models.append(kwargs["model"])
        return pd.DataFrame(
            {
                "instruction_index": [1, 2],
                "completion_turn_1": ["Gen A1", "Gen A2"],
                "completion_turn_2": ["Gen B1", "Gen B2"],
            }
        )

    monkeypatch.setattr(mt_bench_utils, "generate_multiturn", fake_generate_multiturn)
    monkeypatch.setattr(
        mt_bench_utils,
        "load_mt_bench_model_answers",
        lambda model, n_instructions=None: (
            pd.DataFrame(
                {
                    "instruction_index": [2, 1],
                    "completion_turn_1": ["Base A2", "Base A1"],
                    "completion_turn_2": ["Base B2", "Base B1"],
                }
            )
            if model == "gpt-4"
            else None
        ),
    )

    args = CliArgs(
        task="mt-bench",
        model_A="VLLM/example/model-a",
        model_B="gpt-4",
        judge_model="Dummy/J",
        n_instructions=2,
        engine_kwargs={"gpu_memory_utilization": 0.7},
    )

    completions_a, completions_b = mt_bench_utils._generate_mt_bench_completions(
        args=args,
        questions_df=questions_df,
        ignore_cache=False,
    )

    assert generated_models == ["VLLM/example/model-a"]
    assert completions_a.loc[1, "completion_turn_1"] == "Gen A1"
    assert completions_b.loc[1, "completion_turn_1"] == "Base A1"
    assert completions_b.loc[2, "completion_turn_2"] == "Base B2"


def test_run_mt_bench_resolves_native_baseline_and_judge_controls(
    monkeypatch, tmp_path
):
    questions_df = pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )
    captured = {}

    monkeypatch.setattr(
        mt_bench_utils,
        "load_instructions",
        lambda dataset, n_instructions=None: questions_df,
    )
    monkeypatch.setattr(
        mt_bench_utils,
        "_generate_mt_bench_completions",
        lambda args, questions_df, ignore_cache: (
            pd.DataFrame(
                {"completion_turn_1": ["A1"], "completion_turn_2": ["A2"]},
                index=questions_df.index,
            ),
            pd.DataFrame(
                {"completion_turn_1": ["B1"], "completion_turn_2": ["B2"]},
                index=questions_df.index,
            ),
        ),
    )

    def fake_make_model(**kwargs):
        captured["make_model"] = kwargs
        return object()

    monkeypatch.setattr(mt_bench_utils, "make_model", fake_make_model)

    def fake_run_mt_bench_fastchat(**kwargs):
        captured["fastchat"] = kwargs
        return pd.Series([0.0], dtype=float)

    monkeypatch.setattr(
        mt_bench_utils,
        "_run_mt_bench_fastchat",
        fake_run_mt_bench_fastchat,
    )

    args = CliArgs(
        task="mt-bench",
        model_A="VLLM/example/model-a",
        model_B=None,
        judge_model="VLLM/Judge",
        n_instructions=1,
        truncate_judge_input_chars=80000,
        max_judge_model_len=65536,
        mt_bench_judge_mode="fastchat_original",
        engine_kwargs={"tensor_parallel_size": 1},
        judge_engine_kwargs={"tensor_parallel_size": 4},
        result_folder=str(tmp_path),
    )

    mt_bench_utils.run_mt_bench(
        args,
        ignore_cache=False,
        res_folder=tmp_path,
        result_name="mt-bench-test",
    )

    assert args.model_B == "gpt-4"
    assert captured["make_model"]["max_model_len"] == 65536
    assert captured["make_model"]["tensor_parallel_size"] == 4
    assert captured["fastchat"]["args"].truncate_judge_input_chars == 80000
