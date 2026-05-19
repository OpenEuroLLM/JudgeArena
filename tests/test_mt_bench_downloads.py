import importlib
from types import SimpleNamespace

import pandas as pd

import judgearena.instruction_dataset.mt_bench as mt_bench
import judgearena.mt_bench.mt_bench_utils as mt_bench_utils
import judgearena.utils as utils
from judgearena.cli_common import BaseCliArgs


def _mt_bench_args(
    *,
    dataset: str,
    model_A: str,
    model_B: str,
    use_tqdm: bool = False,
    **base_overrides,
) -> BaseCliArgs:
    """Construct a ``BaseCliArgs`` with MT-Bench-specific extras attached."""
    args = BaseCliArgs(**base_overrides)
    args.task = dataset
    args.model_A = model_A
    args.model_B = model_B
    args.use_tqdm = use_tqdm
    return args


def _single_mt_bench_question_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"turn_1": ["Q1"], "turn_2": ["Q1b"]},
        index=pd.Index([1], name="instruction_index"),
    )


def _mt_bench_completion_pair(
    questions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            {
                "completion_turn_1": ["A1"],
                "completion_turn_2": ["A2"],
            },
            index=questions_df.index,
        ),
        pd.DataFrame(
            {
                "completion_turn_1": ["B1"],
                "completion_turn_2": ["B2"],
            },
            index=questions_df.index,
        ),
    )


def _stub_run_mt_bench_inputs(monkeypatch, questions_df: pd.DataFrame) -> None:
    monkeypatch.setattr(
        mt_bench_utils,
        "load_instructions",
        lambda dataset, n_instructions=None: questions_df,
    )
    monkeypatch.setattr(
        mt_bench_utils,
        "_generate_mt_bench_completions",
        lambda args, questions_df, ignore_cache, limit_event_tracker: (
            _mt_bench_completion_pair(questions_df)
        ),
    )


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
    generation_kwargs = []

    monkeypatch.setattr(
        mt_bench_utils, "cache_function_dataframe", lambda fun, **_kwargs: fun()
    )

    def fake_generate_multiturn(
        *,
        questions,
        model,
        truncate_input_chars,
        max_tokens,
        use_tqdm,
        max_model_len,
        chat_template,
        temperature_config,
        limit_event_tracker,
        **engine_kwargs,
    ):
        generated_models.append(model)
        generation_kwargs.append(engine_kwargs)
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

    args = SimpleNamespace(
        model_A="VLLM/example/model-a",
        model_B="gpt-4",
        n_instructions=2,
        truncate_all_input_chars=8192,
        max_out_tokens_models=1024,
        use_tqdm=False,
        max_model_len=16384,
        chat_template=None,
        engine_kwargs={"gpu_memory_utilization": 0.7, "language_model_only": True},
    )

    completions_a, completions_b = mt_bench_utils._generate_mt_bench_completions(
        args=args,
        questions_df=questions_df,
        ignore_cache=False,
        limit_event_tracker=None,
    )

    assert generated_models == ["VLLM/example/model-a"]
    assert generation_kwargs == [
        {"gpu_memory_utilization": 0.7, "language_model_only": True}
    ]
    assert completions_a.loc[1, "completion_turn_1"] == "Gen A1"
    assert completions_b.loc[1, "completion_turn_1"] == "Base A1"
    assert completions_b.loc[2, "completion_turn_2"] == "Base B2"


def test_preset_judging_preflights_token_budget_for_default_mode(monkeypatch):
    preset_judging = importlib.import_module("judgearena.mt_bench.preset_judging")
    tracker = utils.LimitEventTracker()
    captured = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=True):
            text = "".join(
                str(
                    message["content"] if isinstance(message, dict) else message.content
                )
                for message in messages
            )
            return [0] * len(text)

        def encode(self, text):
            return [0] * len(text)

    def fake_do_inference(*, inputs, **kwargs):
        captured["inputs"] = inputs
        return ["score_A: 8\nscore_B: 4"]

    monkeypatch.setattr(preset_judging, "do_inference", fake_do_inference)

    prefs, annotations, metadata, num_inconsistent = (
        preset_judging.judge_mt_bench_with_preset(
            judge_chat_model=object(),
            judge_model="Dummy/J",
            questions=pd.DataFrame(
                {
                    "category": ["writing"],
                    "turn_1": ["Question"],
                    "turn_2": [""],
                    "reference_turn_1": [""],
                    "reference_turn_2": [""],
                },
                index=pd.Index([1], name="question_id"),
            ),
            completions_a=pd.DataFrame(
                {"completion_turn_1": ["A" * 1200], "completion_turn_2": [""]},
                index=pd.Index([1], name="question_id"),
            ),
            completions_b=pd.DataFrame(
                {"completion_turn_1": ["B" * 1200], "completion_turn_2": [""]},
                index=pd.Index([1], name="question_id"),
            ),
            model_a="Model/A",
            model_b="Model/B",
            turns_mode="single",
            swap_mode="fixed",
            truncate_input_chars=None,
            use_tqdm=False,
            prompt_preset="default",
            provide_explanation=False,
            limit_event_tracker=tracker,
            judge_tokenizer=FakeTokenizer(),
            max_judge_model_len=2300,
            max_out_tokens_judge=32,
        )
    )

    prompt_value = captured["inputs"][0]
    assert len(prefs) == 1
    assert len(annotations) == 1
    assert len(metadata) == 1
    assert num_inconsistent == 0
    assert len(FakeTokenizer().apply_chat_template(prompt_value.to_messages())) <= 2012
    assert annotations[0]["answer_a_1_truncated"] is True
    assert annotations[0]["answer_b_1_truncated"] is True
    assert (
        tracker.build_summary()["counts_by_kind"]["judge_input_token_truncation"] >= 1
    )


def test_run_mt_bench_forwards_engine_kwargs_to_judge(monkeypatch, caplog):
    questions_df = _single_mt_bench_question_df()
    captured = {}

    _stub_run_mt_bench_inputs(monkeypatch, questions_df)

    def fake_make_model(
        *,
        model,
        max_tokens,
        temperature=None,
        max_model_len,
        chat_template,
        **kwargs,
    ):
        captured["make_model"] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "max_model_len": max_model_len,
            "chat_template": chat_template,
            "kwargs": kwargs,
        }
        return object()

    monkeypatch.setattr(mt_bench_utils, "make_model", fake_make_model)

    def fake_run_mt_bench_preset(**kwargs):
        captured["run_mt_bench_preset"] = kwargs
        return pd.Series(
            kwargs["questions_df"].index.to_list(),
            dtype=float,
        )

    monkeypatch.setattr(
        mt_bench_utils,
        "_run_mt_bench_preset",
        fake_run_mt_bench_preset,
    )

    args = _mt_bench_args(
        dataset="mt-bench",
        model_A="VLLM/example/model-a",
        model_B="gpt-4",
        judge_model="VLLM/Qwen/Qwen3.5-27B-FP8",
        n_instructions=1,
        truncate_all_input_chars=8192,
        max_out_tokens_models=1024,
        max_out_tokens_judge=256,
        max_model_len=16384,
        chat_template=None,
        provide_explanation=False,
        swap_mode="fixed",
        judge_prompt_preset="default",
        engine_kwargs={"gpu_memory_utilization": 0.7, "language_model_only": True},
    )

    caplog.set_level("WARNING", logger=mt_bench_utils.__name__)
    mt_bench_utils.run_mt_bench(args, ignore_cache=False)

    assert captured["make_model"]["max_tokens"] == 256
    assert captured["make_model"]["max_model_len"] is None
    assert captured["make_model"]["kwargs"]["limit_event_stage"] == "judge_model_init"
    assert (
        captured["make_model"]["kwargs"]["limit_event_model_spec"]
        == "VLLM/Qwen/Qwen3.5-27B-FP8"
    )
    assert "limit_event_tracker" in captured["make_model"]["kwargs"]
    assert captured["run_mt_bench_preset"]["prompt_preset"] == "default"
    assert "MT-Bench ignores provide_explanation=False" not in caplog.text


def test_run_mt_bench_default_respects_judge_temperature_from_engine_kwargs(
    monkeypatch,
):
    questions_df = _single_mt_bench_question_df()
    captured = {}

    _stub_run_mt_bench_inputs(monkeypatch, questions_df)

    def fake_make_model(
        *,
        model,
        max_tokens,
        temperature=None,
        max_model_len,
        chat_template,
        **kwargs,
    ):
        captured["temperature"] = temperature
        return object()

    monkeypatch.setattr(mt_bench_utils, "make_model", fake_make_model)
    monkeypatch.setattr(
        mt_bench_utils,
        "_run_mt_bench_preset",
        lambda **kwargs: pd.Series([0.0], dtype=float),
    )

    args = _mt_bench_args(
        dataset="mt-bench",
        model_A="VLLM/example/model-a",
        model_B="gpt-4",
        judge_model="VLLM/Qwen/Qwen3.5-27B-FP8",
        n_instructions=1,
        max_out_tokens_judge=256,
        engine_kwargs={"temperature": 0.8},
        judge_engine_kwargs={"temperature": 0.4},
    )

    mt_bench_utils.run_mt_bench(args, ignore_cache=False)

    assert captured["temperature"] == 0.4
