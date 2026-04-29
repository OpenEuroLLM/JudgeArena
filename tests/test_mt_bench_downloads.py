import importlib
from datetime import UTC, datetime
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


def test_mt_bench_prompt_templates_preserve_multi_turn_reference_blocks():
    prompt_templates = importlib.import_module("judgearena.mt_bench.prompt_templates")

    rendered = prompt_templates.build_mt_bench_user_prompt_template(
        multi_turn=True,
        ref_based=True,
    )

    assert "<|The Start of Reference Answer|>" in rendered
    assert "### User:\n{question_1}" in rendered
    assert "### Reference answer:\n{ref_answer_2}" in rendered
    assert "### Assistant A:\n{answer_a_2}" in rendered
    assert "### Assistant B:\n{answer_b_2}" in rendered


def test_select_preset_prompt_uses_default_score_mode_with_mt_bench_template():
    preset_judging = importlib.import_module("judgearena.mt_bench.preset_judging")

    prompt = preset_judging._select_preset_prompt(
        "math",
        multi_turn=True,
        prompt_preset="default",
        provide_explanation=False,
    )

    assert prompt.parser_mode == "score"
    assert prompt.ref_based is True
    assert prompt.multi_turn is True
    assert prompt.system_prompt
    assert "<|The Start of Reference Answer|>" in prompt.user_prompt_template
    assert "### Assistant A:\n{answer_a_2}" in prompt.user_prompt_template


def test_select_preset_prompt_uses_skywork_verdict_mode_with_mt_bench_template():
    preset_judging = importlib.import_module("judgearena.mt_bench.preset_judging")

    prompt = preset_judging._select_preset_prompt(
        "writing",
        multi_turn=True,
        prompt_preset=SKYWORK_JUDGE_PROMPT_PRESET,
        provide_explanation=True,
    )

    assert prompt.parser_mode == "verdict"
    assert prompt.ref_based is False
    assert prompt.multi_turn is True
    assert prompt.system_prompt is None
    assert "Please briefly explain your reasoning first" in prompt.user_prompt_template
    assert "### Assistant B:\n{answer_b_2}" in prompt.user_prompt_template


def test_preset_judging_uses_shared_swap_mode_both_semantics(monkeypatch):
    preset_judging = importlib.import_module("judgearena.mt_bench.preset_judging")
    questions_df = pd.DataFrame(
        {
            "category": ["writing"],
            "turn_1": ["Q1"],
            "turn_2": ["Q2"],
            "reference_turn_1": [""],
            "reference_turn_2": [""],
        },
        index=pd.Index([1], name="question_id"),
    )
    completions_a = pd.DataFrame(
        {
            "completion_turn_1": ["A1"],
            "completion_turn_2": ["A2"],
        },
        index=questions_df.index,
    )
    completions_b = pd.DataFrame(
        {
            "completion_turn_1": ["B1"],
            "completion_turn_2": ["B2"],
        },
        index=questions_df.index,
    )
    call_count = {"count": 0}

    def fake_do_inference(**kwargs):
        call_count["count"] += 1
        if call_count["count"] <= 2:
            return ["score_A: 9\nscore_B: 3"]
        return ["score_A: 2\nscore_B: 7"]

    monkeypatch.setattr(preset_judging, "do_inference", fake_do_inference)

    prefs, annotations, metadata, num_inconsistent = (
        preset_judging.judge_mt_bench_with_preset(
            judge_chat_model=object(),
            judge_model="Dummy/J",
            questions=questions_df,
            completions_a=completions_a,
            completions_b=completions_b,
            model_a="Model/A",
            model_b="Model/B",
            turns_mode="both",
            swap_mode="both",
            truncate_input_chars=None,
            use_tqdm=False,
            prompt_preset="default",
            provide_explanation=False,
        )
    )

    assert len(prefs) == 4
    assert len(annotations) == 4
    assert len(metadata) == 4
    assert num_inconsistent == 0
    assert [row["swapped"] for row in annotations] == [False, False, True, True]


def test_preset_judging_uses_shared_char_truncation_event_kind(monkeypatch):
    preset_judging = importlib.import_module("judgearena.mt_bench.preset_judging")
    tracker = utils.LimitEventTracker()

    monkeypatch.setattr(
        preset_judging,
        "do_inference",
        lambda **kwargs: ["score_A: 8\nscore_B: 4"],
    )

    prefs, annotations, metadata, num_inconsistent = (
        preset_judging.judge_mt_bench_with_preset(
            judge_chat_model=object(),
            judge_model="Dummy/J",
            questions=pd.DataFrame(
                {
                    "category": ["writing"],
                    "turn_1": ["Q" * 20],
                    "turn_2": [""],
                    "reference_turn_1": [""],
                    "reference_turn_2": [""],
                },
                index=pd.Index([1], name="question_id"),
            ),
            completions_a=pd.DataFrame(
                {"completion_turn_1": ["A" * 30], "completion_turn_2": [""]},
                index=pd.Index([1], name="question_id"),
            ),
            completions_b=pd.DataFrame(
                {"completion_turn_1": ["B" * 30], "completion_turn_2": [""]},
                index=pd.Index([1], name="question_id"),
            ),
            model_a="Model/A",
            model_b="Model/B",
            turns_mode="single",
            swap_mode="fixed",
            truncate_input_chars=10,
            use_tqdm=False,
            prompt_preset="default",
            provide_explanation=False,
            limit_event_tracker=tracker,
        )
    )

    assert len(prefs) == 1
    assert len(annotations) == 1
    assert len(metadata) == 1
    assert num_inconsistent == 0
    assert tracker.build_summary()["counts_by_kind"]["judge_input_char_truncation"] == 3


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
    assert captured["make_model"]["temperature"] is None
    assert captured["run_mt_bench_preset"]["args"].swap_mode == "fixed"
    assert captured["run_mt_bench_preset"]["prompt_preset"] == "default"
    assert (
        captured["run_mt_bench_preset"]["args"].strip_thinking_before_judging is False
    )
    assert "MT-Bench ignores provide_explanation=False" not in caplog.text


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

    def fake_run_mt_bench_preset(**kwargs):
        captured["kwargs"] = kwargs
        return pd.Series([0.0], dtype=float)

    monkeypatch.setattr(
        mt_bench_utils,
        "_run_mt_bench_preset",
        fake_run_mt_bench_preset,
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


def test_run_mt_bench_default_respects_judge_temperature_from_engine_kwargs(
    monkeypatch,
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


def test_save_mt_bench_results_uses_explicit_res_folder(tmp_path, monkeypatch):
    captured = {}

    def fake_write_run_metadata(**kwargs):
        captured["output_dir"] = kwargs["output_dir"]
        return kwargs["output_dir"] / "run-metadata.v1.json"

    monkeypatch.setattr(mt_bench_utils, "write_run_metadata", fake_write_run_metadata)

    args = _mt_bench_args(
        dataset="mt-bench",
        model_A="VLLM/example/model-a",
        model_B="gpt-4",
        judge_model="VLLM/Qwen/Qwen3.5-27B-FP8",
        result_folder=str(tmp_path / "results-root"),
    )
    explicit_res_folder = tmp_path / "explicit-run"
    result_name = "mt-bench-run"

    mt_bench_utils._save_mt_bench_results(
        args=args,
        res_folder=explicit_res_folder,
        result_name=result_name,
        results={"task": "mt-bench"},
        annotations_df=pd.DataFrame([{"question_id": 1, "turn": 1}]),
        questions_df=pd.DataFrame(
            {"turn_1": ["Q1"], "turn_2": ["Q2"]},
            index=pd.Index([1], name="question_id"),
        ),
        pricing_reference=None,
        started_at_utc=datetime.now(UTC),
    )

    assert captured["output_dir"] == explicit_res_folder
    assert (explicit_res_folder / f"args-{result_name}.json").exists()
    assert (explicit_res_folder / f"results-{result_name}.json").exists()
    assert (explicit_res_folder / f"{result_name}-annotations.csv").exists()
    assert not (tmp_path / "results-root" / result_name).exists()
