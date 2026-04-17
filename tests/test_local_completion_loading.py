import pandas as pd

import judgearena.evaluate as evaluate
import judgearena.generate_and_evaluate as generate_and_evaluate
from judgearena.cli_common import parse_optional_bool
from judgearena.generate_and_evaluate import CliArgs
from judgearena.generate_and_evaluate import main as main_generate_and_eval
from judgearena.judge_prompt_presets import SKYWORK_JUDGE_PROMPT_PRESET


def test_load_judge_prompt_without_explanation_uses_freeform_scores():
    _system_prompt, user_prompt = evaluate.load_judge_system_and_user_prompt(
        provide_explanation=False
    )

    assert "valid JSON" not in user_prompt
    assert "score_A:" in user_prompt
    assert "score_B:" in user_prompt
    assert "Assistant A's Answer" in user_prompt


def test_load_judge_prompt_with_explanation_uses_freeform_scores():
    _system_prompt, user_prompt = evaluate.load_judge_system_and_user_prompt(
        provide_explanation=True
    )

    assert "valid JSON" not in user_prompt
    assert "first starts with an explanation of your judgement" in user_prompt
    assert "score_A:" in user_prompt
    assert "score_B:" in user_prompt
    assert "Assistant B's Answer" in user_prompt


def test_parse_optional_bool_accepts_explicit_true_false_values():
    assert parse_optional_bool(None) is True
    assert parse_optional_bool("true") is True
    assert parse_optional_bool("False") is False


def test_main_passes_qwen_defaults_and_aligns_dataset_completions(
    tmp_path, monkeypatch
):
    instructions = pd.DataFrame(
        {"instruction": ["Instruction B", "Instruction A"]},
        index=pd.Index(["b", "a"], name="instruction_index"),
    )
    captured = {}

    monkeypatch.setattr(
        generate_and_evaluate,
        "load_instructions",
        lambda dataset, n_instructions=None: instructions,
    )

    def fake_try_load_dataset_completions(dataset, model, n_instructions):
        if model == "Dummy/model-a":
            return pd.DataFrame(
                {
                    "instruction_index": ["a", "b"],
                    "completion": ["Answer A", "no answer"],
                }
            )
        return pd.DataFrame(
            {
                "instruction_index": ["a", "b"],
                "completion": ["Answer B", "Answer C"],
            }
        )

    monkeypatch.setattr(
        generate_and_evaluate,
        "try_load_dataset_completions",
        fake_try_load_dataset_completions,
    )

    def fake_make_model(*, model, max_tokens, max_model_len, chat_template, **kwargs):
        captured["make_model"] = {
            "model": model,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "chat_template": chat_template,
            "kwargs": kwargs,
        }
        return object()

    def fake_judge_and_parse_prefs(**kwargs):
        captured["judge_kwargs"] = kwargs
        annotations = [{"judge_completion": "score_A: 0\nscore_B: 10"}] * len(
            kwargs["instructions"]
        )
        return annotations, None, pd.Series([1.0] * len(kwargs["instructions"]))

    monkeypatch.setattr(generate_and_evaluate, "make_model", fake_make_model)
    monkeypatch.setattr(
        generate_and_evaluate,
        "judge_and_parse_prefs",
        fake_judge_and_parse_prefs,
    )

    prefs = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/model-a",
            model_B="Dummy/model-b",
            judge_model="VLLM/Qwen/Qwen3.5-27B-FP8",
            n_instructions=2,
            result_folder=str(tmp_path / "results"),
        )
    )

    assert prefs.tolist() == [1.0, 1.0]
    assert captured["make_model"]["kwargs"]["thinking_token_budget"] == 512
    assert captured["make_model"]["kwargs"]["limit_event_stage"] == "judge_model_init"
    assert captured["make_model"]["kwargs"]["limit_event_model_spec"] == (
        "VLLM/Qwen/Qwen3.5-27B-FP8"
    )
    assert captured["judge_kwargs"]["instructions"] == [
        "Instruction B",
        "Instruction A",
    ]
    assert captured["judge_kwargs"]["completions_A"] == ["no answer", "Answer A"]
    assert captured["judge_kwargs"]["completions_B"] == ["Answer C", "Answer B"]
    assert captured["judge_kwargs"]["case_ids"] == ["b", "a"]
    assert captured["judge_kwargs"]["prompt_preset"] == "default"
    assert captured["judge_kwargs"]["parser_mode"] == "score"
    assert captured["judge_kwargs"]["strip_thinking_before_judging"] is False


def test_main_does_not_pass_thinking_budget_to_non_reasoning_vllm_judge(
    tmp_path, monkeypatch
):
    instructions = pd.DataFrame(
        {"instruction": ["Instruction A"]},
        index=pd.Index([1], name="instruction_index"),
    )
    completions_df = pd.DataFrame(
        {"instruction_index": [1], "completion": ["Loaded answer"]}
    )
    captured = {}

    monkeypatch.setattr(
        generate_and_evaluate,
        "load_instructions",
        lambda dataset, n_instructions=None: instructions,
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "try_load_dataset_completions",
        lambda dataset, model, n_instructions: completions_df,
    )

    def fake_make_model(*, model, max_tokens, max_model_len, chat_template, **kwargs):
        captured["make_model"] = kwargs
        return object()

    monkeypatch.setattr(generate_and_evaluate, "make_model", fake_make_model)
    monkeypatch.setattr(
        generate_and_evaluate,
        "judge_and_parse_prefs",
        lambda **kwargs: (
            [{"judge_completion": "score_A: 1\nscore_B: 2"}],
            None,
            pd.Series([1.0]),
        ),
    )

    prefs = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/model-a",
            model_B="Dummy/model-b",
            judge_model="VLLM/meta-llama/Llama-3.3-70B-Instruct",
            n_instructions=1,
            result_folder=str(tmp_path / "results"),
        )
    )

    assert prefs.tolist() == [1.0]
    assert "thinking_token_budget" not in captured["make_model"]
    assert captured["make_model"]["limit_event_stage"] == "judge_model_init"


def test_main_preserves_explicit_reasoning_engine_kwargs_for_non_qwen_vllm_judge(
    tmp_path, monkeypatch
):
    instructions = pd.DataFrame(
        {"instruction": ["Instruction A"]},
        index=pd.Index([1], name="instruction_index"),
    )
    completions_df = pd.DataFrame(
        {"instruction_index": [1], "completion": ["Loaded answer"]}
    )
    captured = {}

    monkeypatch.setattr(
        generate_and_evaluate,
        "load_instructions",
        lambda dataset, n_instructions=None: instructions,
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "try_load_dataset_completions",
        lambda dataset, model, n_instructions: completions_df,
    )

    def fake_make_model(*, model, max_tokens, max_model_len, chat_template, **kwargs):
        captured["make_model"] = kwargs
        return object()

    monkeypatch.setattr(generate_and_evaluate, "make_model", fake_make_model)
    monkeypatch.setattr(
        generate_and_evaluate,
        "judge_and_parse_prefs",
        lambda **kwargs: (
            [{"judge_completion": "score_A: 1\nscore_B: 2"}],
            None,
            pd.Series([1.0]),
        ),
    )

    prefs = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/model-a",
            model_B="Dummy/model-b",
            judge_model="VLLM/meta-llama/Llama-3.3-70B-Instruct",
            n_instructions=1,
            result_folder=str(tmp_path / "results"),
            engine_kwargs={
                "reasoning_parser": "custom-parser",
                "thinking_token_budget": 2048,
            },
        )
    )

    assert prefs.tolist() == [1.0]
    assert captured["make_model"]["reasoning_parser"] == "custom-parser"
    assert captured["make_model"]["thinking_token_budget"] == 2048


def test_annotate_battles_warns_when_judge_inputs_are_truncated(monkeypatch, capsys):
    captured = {}

    def fake_do_inference(
        *,
        chat_model,
        inputs,
        use_tqdm,
        usage_tracker,
        usage_phase,
        usage_model_spec,
    ):
        captured["judge_prompt"] = inputs[0].to_messages()[1].content
        return ["score_A: 0\nscore_B: 10"]

    monkeypatch.setattr(evaluate, "do_inference", fake_do_inference)

    annotations = evaluate.annotate_battles(
        judge_chat_model=object(),
        instructions=["Instruction"],
        completions_A=["Answer A"],
        completions_B=["Answer B"],
        truncate_input_chars=3,
    )

    stdout = capsys.readouterr().out
    assert (
        "Warning: truncated 2 judge inputs to 3 characters before evaluation." in stdout
    )
    assert "Ans" in captured["judge_prompt"]
    assert "Answer A" not in captured["judge_prompt"]
    assert "Answer B" not in captured["judge_prompt"]
    assert "valid JSON" not in captured["judge_prompt"]
    assert "score_A:" in captured["judge_prompt"]
    assert annotations[0].completion_A == "Answer A"
    assert annotations[0].completion_B == "Answer B"


def test_resolve_judge_prompts_supports_optional_skywork_preset():
    resolved = evaluate.resolve_judge_prompts(
        provide_explanation=False,
        prompt_preset=SKYWORK_JUDGE_PROMPT_PRESET,
    )

    assert resolved.preset_name == SKYWORK_JUDGE_PROMPT_PRESET
    assert resolved.parser_mode == "verdict"
    assert resolved.system_prompt is None
    assert "[[A]]" in resolved.user_prompt_template
    assert "score_A:" not in resolved.user_prompt_template
    assert "[User Question]" in resolved.user_prompt_template
    assert "Assistant A's Answer" in resolved.user_prompt_template


def test_resolve_judge_prompts_skywork_explanation_prompt_has_fixed_answer_labels():
    resolved = evaluate.resolve_judge_prompts(
        provide_explanation=True,
        prompt_preset=SKYWORK_JUDGE_PROMPT_PRESET,
    )

    assert (
        "Please briefly explain your reasoning first" in resolved.user_prompt_template
    )
    assert "Assistant B's Answer" in resolved.user_prompt_template


def test_annotate_battles_records_limit_events_for_stripping_and_truncation(
    monkeypatch,
):
    tracker = evaluate.LimitEventTracker()

    monkeypatch.setattr(
        evaluate,
        "do_inference",
        lambda **kwargs: ["[[A]]"],
    )

    evaluate.annotate_battles(
        judge_chat_model=object(),
        instructions=["Instruction"],
        completions_A=["<think>hidden</think>Visible answer"],
        completions_B=["Short"],
        case_ids=["case-1"],
        truncate_input_chars=5,
        strip_thinking_before_judging=True,
        limit_event_tracker=tracker,
        prompt_preset=SKYWORK_JUDGE_PROMPT_PRESET,
    )

    summary = tracker.build_summary()

    assert summary["counts_by_kind"]["thinking_trace_stripped_before_judging"] == 1
    assert summary["counts_by_kind"]["judge_input_char_truncation"] == 1


def test_main_passes_qwen_only_battle_budget_and_prompt_preset(tmp_path, monkeypatch):
    instructions = pd.DataFrame(
        {"instruction": ["Instruction A"]},
        index=pd.Index([1], name="instruction_index"),
    )
    captured = {}

    monkeypatch.setattr(
        generate_and_evaluate,
        "load_instructions",
        lambda dataset, n_instructions=None: instructions,
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "try_load_dataset_completions",
        lambda dataset, model, n_instructions: None,
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "cache_function_dataframe",
        lambda fun, **_kwargs: fun(),
    )

    def fake_generate_instructions(
        *,
        instructions,
        model,
        truncate_input_chars,
        max_tokens,
        max_model_len,
        chat_template,
        use_tqdm,
        usage_tracker,
        usage_phase,
        limit_event_tracker,
        **engine_kwargs,
    ):
        captured.setdefault("generation_calls", []).append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "engine_kwargs": engine_kwargs,
            }
        )
        return pd.DataFrame(
            {
                "instruction_index": [1],
                "completion": [f"{model}-answer"],
                "generation_prompt_truncated": [False],
                "generation_output_finish_reason": [None],
                "generation_output_hit_token_limit": [False],
            }
        )

    monkeypatch.setattr(
        generate_and_evaluate, "generate_instructions", fake_generate_instructions
    )

    def fake_judge_and_parse_prefs(**kwargs):
        captured["judge_kwargs"] = kwargs
        return [{"judge_completion": "[[A]]"}], None, pd.Series([0.0])

    monkeypatch.setattr(
        generate_and_evaluate,
        "judge_and_parse_prefs",
        fake_judge_and_parse_prefs,
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "make_model",
        lambda **kwargs: object(),
    )

    prefs = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="VLLM/Qwen/Qwen3.5-27B-FP8",
            model_B="VLLM/allenai/Olmo-3-7B-Instruct",
            judge_model="Dummy/judge",
            n_instructions=1,
            judge_prompt_preset=SKYWORK_JUDGE_PROMPT_PRESET,
            battle_thinking_token_budget=512,
            strip_thinking_before_judging=True,
            result_folder=str(tmp_path / "results"),
        )
    )

    assert prefs.tolist() == [0.0]
    assert len(captured["generation_calls"]) == 2
    assert (
        captured["generation_calls"][0]["engine_kwargs"]["thinking_token_budget"] == 512
    )
    assert (
        "thinking_token_budget" not in captured["generation_calls"][1]["engine_kwargs"]
    )
    assert captured["judge_kwargs"]["prompt_preset"] == SKYWORK_JUDGE_PROMPT_PRESET
    assert captured["judge_kwargs"]["parser_mode"] == "verdict"
    assert captured["judge_kwargs"]["strip_thinking_before_judging"] is True


def test_generation_cache_name_changes_with_generation_settings():
    args = CliArgs(
        dataset="alpaca-eval",
        model_A="Dummy/model-a",
        model_B="Dummy/model-b",
        judge_model="Dummy/judge",
        n_instructions=1,
        max_out_tokens_models=1024,
        battle_thinking_token_budget=256,
    )
    changed_args = CliArgs(
        dataset="alpaca-eval",
        model_A="Dummy/model-a",
        model_B="Dummy/model-b",
        judge_model="Dummy/judge",
        n_instructions=1,
        max_out_tokens_models=4096,
        battle_thinking_token_budget=512,
    )

    cache_name = generate_and_evaluate._generation_cache_name(
        args, model_spec="VLLM/Qwen/Qwen3.5-27B-FP8"
    )
    changed_cache_name = generate_and_evaluate._generation_cache_name(
        changed_args, model_spec="VLLM/Qwen/Qwen3.5-27B-FP8"
    )

    assert cache_name != changed_cache_name
