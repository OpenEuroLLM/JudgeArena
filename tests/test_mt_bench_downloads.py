from types import SimpleNamespace

import pandas as pd

import judgearena.instruction_dataset.mt_bench as mt_bench
import judgearena.mt_bench.fastchat_compat as fastchat_compat
import judgearena.mt_bench.mt_bench_utils as mt_bench_utils
import judgearena.utils as utils
from judgearena.cli_common import BaseCliArgs
from judgearena.judge_prompt_presets import SKYWORK_JUDGE_PROMPT_PRESET


def _mt_bench_args(
    *,
    dataset: str,
    model_A: str,
    model_B: str,
    use_tqdm: bool = False,
    **base_overrides,
) -> BaseCliArgs:
    """Construct a ``BaseCliArgs`` with MT-Bench CLI-style extras attached.

    Using the real dataclass here keeps tests close to the production CLI
    contract while attaching the task/model fields owned by ``CliArgs``.
    """
    args = BaseCliArgs(**base_overrides)
    args.task = dataset
    args.model_A = model_A
    args.model_B = model_B
    args.use_tqdm = use_tqdm
    return args


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
    generation_strip_flags = []

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
        usage_tracker,
        usage_phase,
        limit_event_tracker,
        strip_thinking_before_turn_2_prompt,
        **engine_kwargs,
    ):
        generated_models.append(model)
        generation_kwargs.append(engine_kwargs)
        generation_strip_flags.append(strip_thinking_before_turn_2_prompt)
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
        battle_thinking_token_budget=None,
        strip_thinking_before_judging=True,
        engine_kwargs={"gpu_memory_utilization": 0.7, "language_model_only": True},
    )

    completions_a, completions_b = mt_bench_utils._generate_mt_bench_completions(
        args=args,
        questions_df=questions_df,
        ignore_cache=False,
        usage_tracker=object(),
        limit_event_tracker=None,
    )

    assert generated_models == ["VLLM/example/model-a"]
    assert generation_kwargs == [
        {"gpu_memory_utilization": 0.7, "language_model_only": True}
    ]
    assert generation_strip_flags == [True]
    assert completions_a.loc[1, "completion_turn_1"] == "Gen A1"
    assert completions_b.loc[1, "completion_turn_1"] == "Base A1"
    assert completions_b.loc[2, "completion_turn_2"] == "Base B2"


def test_parse_fastchat_verdict_accepts_bracketed_verdicts_after_thinking():
    assert (
        fastchat_compat._parse_fastchat_verdict(
            "<think>Need a longer chain.</think>[[A]]"
        )
        == "A"
    )
    assert fastchat_compat._parse_fastchat_verdict("[[B]]") == "B"
    assert fastchat_compat._parse_fastchat_verdict("[[C]]") == "tie"


def test_parse_fastchat_verdict_marks_non_bracketed_outputs_as_error():
    assert fastchat_compat._parse_fastchat_verdict("A") == "error"
    assert fastchat_compat._parse_fastchat_verdict('{"verdict":"B"}') == "error"


def test_pair_v2_system_prompt_matches_original_fastchat_contract():
    rendered = fastchat_compat._PAIR_V2.system_prompt

    assert "provide a short explanation" in rendered
    assert "valid JSON" not in rendered
    assert '"[[A]]"' in rendered
    assert '"[[B]]"' in rendered
    assert '"[[C]]"' in rendered
    assert rendered.endswith(
        'After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.\n'
    )


def test_conservative_winner_marks_one_sided_parse_failures_as_error():
    assert fastchat_compat._conservative_winner("model_A", "error") == (
        "error",
        False,
    )
    assert fastchat_compat._conservative_winner("error", "model_B") == (
        "error",
        False,
    )
    assert fastchat_compat._conservative_winner("error", "error") == ("error", False)
    assert fastchat_compat._conservative_winner("model_A", "model_B") == ("tie", True)


def test_run_mt_bench_forwards_engine_kwargs_to_judge(monkeypatch, caplog):
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
        lambda args, questions_df, ignore_cache, usage_tracker, limit_event_tracker: (
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
        ),
    )

    def fake_make_model(
        *, model, max_tokens, temperature, max_model_len, chat_template, **kwargs
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

    def fake_run_mt_bench_fastchat(**kwargs):
        captured["run_mt_bench_fastchat"] = kwargs
        return pd.Series(
            kwargs["questions_df"].index.to_list(),
            dtype=float,
        )

    monkeypatch.setattr(
        mt_bench_utils,
        "_run_mt_bench_fastchat",
        fake_run_mt_bench_fastchat,
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
        battle_thinking_token_budget=None,
        strip_thinking_before_judging=False,
        engine_kwargs={"gpu_memory_utilization": 0.7, "language_model_only": True},
    )

    caplog.set_level("WARNING", logger=mt_bench_utils.__name__)
    mt_bench_utils.run_mt_bench(args, ignore_cache=False)

    assert args.swap_mode == "fixed"
    assert args.max_out_tokens_judge == 256
    assert args.max_model_len == 16384
    assert args.max_judge_model_len is None
    assert captured["make_model"]["max_tokens"] == 256
    assert captured["make_model"]["max_model_len"] is None
    assert captured["make_model"]["kwargs"] == {
        "gpu_memory_utilization": 0.7,
        "language_model_only": True,
        "thinking_token_budget": 512,
        "kv_cache_dtype": "fp8",
        "limit_event_stage": "judge_model_init",
        "limit_event_model_spec": "VLLM/Qwen/Qwen3.5-27B-FP8",
        "limit_event_tracker": captured["make_model"]["kwargs"]["limit_event_tracker"],
    }
    assert captured["run_mt_bench_fastchat"]["args"].swap_mode == "fixed"
    assert captured["run_mt_bench_fastchat"]["prompt_preset"] == "default"
    assert (
        captured["run_mt_bench_fastchat"]["args"].strip_thinking_before_judging is False
    )
    assert (
        "MT-Bench judge prompts request an explanation before the final verdict"
        in caplog.text
    )
    assert "max_out_tokens_judge=256" in caplog.text


def test_select_prompt_supports_optional_skywork_mt_bench_preset():
    prompt = fastchat_compat._select_prompt(
        "writing",
        multi_turn=False,
        prompt_preset=SKYWORK_JUDGE_PROMPT_PRESET,
    )

    assert prompt.name == "skywork-pair-v2"
    assert prompt.ref_based is False


def test_run_mt_bench_keeps_skywork_prompt_preset(monkeypatch):
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
        lambda args, questions_df, ignore_cache, usage_tracker, limit_event_tracker: (
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
        ),
    )
    monkeypatch.setattr(mt_bench_utils, "make_model", lambda **kwargs: object())

    def fake_run_mt_bench_fastchat(**kwargs):
        captured["kwargs"] = kwargs
        return pd.Series([0.0], dtype=float)

    monkeypatch.setattr(
        mt_bench_utils,
        "_run_mt_bench_fastchat",
        fake_run_mt_bench_fastchat,
    )

    args = _mt_bench_args(
        dataset="mt-bench",
        model_A="VLLM/example/model-a",
        model_B="gpt-4",
        judge_model="VLLM/Skywork/Skywork-Critic-Llama-3.1-8B",
        n_instructions=1,
        truncate_all_input_chars=8192,
        truncate_judge_input_chars=80000,
        max_out_tokens_models=1024,
        max_out_tokens_judge=256,
        max_model_len=16384,
        max_judge_model_len=65536,
        chat_template=None,
        provide_explanation=False,
        swap_mode="both",
        judge_prompt_preset=SKYWORK_JUDGE_PROMPT_PRESET,
        battle_thinking_token_budget=512,
        strip_thinking_before_judging=True,
        engine_kwargs={"gpu_memory_utilization": 0.7, "language_model_only": True},
    )

    mt_bench_utils.run_mt_bench(args, ignore_cache=False)

    assert captured["kwargs"]["prompt_preset"] == SKYWORK_JUDGE_PROMPT_PRESET
    assert captured["kwargs"]["args"].strip_thinking_before_judging is True
    assert args.max_judge_model_len == 65536
    assert args.truncate_judge_input_chars == 80000
    assert captured["kwargs"]["args"].truncate_judge_input_chars == 80000
    assert captured["kwargs"]["args"].max_judge_model_len == 65536
