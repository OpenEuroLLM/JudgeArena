import pandas as pd

import judgearena.evaluate as evaluate
import judgearena.generate_and_evaluate as generate_and_evaluate
from judgearena.generate_and_evaluate import CliArgs
from judgearena.generate_and_evaluate import main as main_generate_and_eval


def test_load_judge_prompt_without_explanation_uses_freeform_scores():
    _system_prompt, user_prompt = evaluate.load_judge_system_and_user_prompt(
        provide_explanation=False
    )

    assert "valid JSON" not in user_prompt
    assert "score_A:" in user_prompt
    assert "score_B:" in user_prompt


def test_load_judge_prompt_with_explanation_uses_freeform_scores():
    _system_prompt, user_prompt = evaluate.load_judge_system_and_user_prompt(
        provide_explanation=True
    )

    assert "valid JSON" not in user_prompt
    assert "first starts with an explanation of your judgement" in user_prompt
    assert "score_A:" in user_prompt
    assert "score_B:" in user_prompt


def test_main_passes_thinking_budget_to_vllm_judge(tmp_path, monkeypatch):
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
        captured["make_model"] = {
            "model": model,
            "max_tokens": max_tokens,
            "max_model_len": max_model_len,
            "chat_template": chat_template,
            "kwargs": kwargs,
        }
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
            judge_model="VLLM/Qwen/Qwen3.5-27B-FP8",
            n_instructions=1,
            result_folder=str(tmp_path / "results"),
        )
    )

    assert prefs.tolist() == [1.0]
    assert "structured_outputs_json" not in captured["make_model"]["kwargs"]
    assert captured["make_model"]["kwargs"]["thinking_token_budget"] == 512


def test_main_passes_thinking_budget_to_vllm_judge_when_explanation_requested(
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
            [{"judge_completion": "Explanation: ok\nscore_A: 1\nscore_B: 2"}],
            None,
            pd.Series([1.0]),
        ),
    )

    prefs = main_generate_and_eval(
        CliArgs(
            dataset="alpaca-eval",
            model_A="Dummy/model-a",
            model_B="Dummy/model-b",
            judge_model="VLLM/Qwen/Qwen3.5-27B-FP8",
            n_instructions=1,
            provide_explanation=True,
            result_folder=str(tmp_path / "results"),
        )
    )

    assert prefs.tolist() == [1.0]
    assert "structured_outputs_json" not in captured["make_model"]
    assert captured["make_model"]["thinking_token_budget"] == 512


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


def test_annotate_battles_warns_when_judge_completions_are_truncated(
    monkeypatch, capsys
):
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
        "Warning: truncated 2 judge completions to 3 characters before evaluation."
        in stdout
    )
    assert "Ans" in captured["judge_prompt"]
    assert "Answer A" not in captured["judge_prompt"]
    assert "Answer B" not in captured["judge_prompt"]
    assert "valid JSON" not in captured["judge_prompt"]
    assert "score_A:" in captured["judge_prompt"]
    assert annotations[0].completion_A == "Answer A"
    assert annotations[0].completion_B == "Answer B"
