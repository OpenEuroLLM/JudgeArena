import pandas as pd

import judgearena.evaluate as evaluate
import judgearena.generate_and_evaluate as generate_and_evaluate
from judgearena.generate_and_evaluate import CliArgs
from judgearena.generate_and_evaluate import main as main_generate_and_eval


def test_build_pair_score_json_schema_covers_valid_range():
    schema = evaluate.build_pair_score_json_schema()

    assert schema["type"] == "object"
    assert set(schema["required"]) == {"score_A", "score_B"}
    for key in ("score_A", "score_B"):
        assert schema["properties"][key]["type"] == "integer"
        assert schema["properties"][key]["minimum"] == 0
        assert schema["properties"][key]["maximum"] == 10
    assert schema["additionalProperties"] is False


def test_main_aligns_local_reference_by_instruction_index(tmp_path, monkeypatch):
    instructions = pd.DataFrame(
        {"instruction": ["Instruction B", "Instruction A"]},
        index=pd.Index(["b", "a"], name="instruction_index"),
    )
    reference_path = tmp_path / "m-arena-hard-en-reference.csv"
    pd.DataFrame(
        {
            "instruction_index": ["a", "b"],
            "output": ["Answer A", "Answer B"],
        }
    ).to_csv(reference_path, index=False)

    monkeypatch.setattr(
        generate_and_evaluate,
        "load_instructions",
        lambda dataset, n_instructions=None: (
            instructions.head(n_instructions)
            if n_instructions is not None
            else instructions
        ),
    )
    monkeypatch.setattr(
        generate_and_evaluate,
        "cache_function_dataframe",
        lambda fun, **_kwargs: fun(),
    )

    captured = {}

    def fake_judge_and_parse_prefs(
        *,
        judge_chat_model,
        instructions,
        completions_A,
        completions_B,
        swap_mode,
        provide_explanation,
        system_prompt,
        user_prompt_template,
        truncate_input_chars,
        use_tqdm,
    ):
        captured["instructions"] = instructions
        captured["completions_A"] = completions_A
        captured["completions_B"] = completions_B
        annotations = [{"judge_completion": "score A: 0 score B: 10"}] * len(
            instructions
        )
        prefs = pd.Series([1.0] * len(instructions))
        return annotations, [], prefs

    monkeypatch.setattr(
        generate_and_evaluate,
        "judge_and_parse_prefs",
        fake_judge_and_parse_prefs,
    )

    prefs = main_generate_and_eval(
        CliArgs(
            dataset="m-arena-hard-en",
            model_A="Dummy/no answer",
            model_B=str(reference_path),
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=2,
            result_folder=str(tmp_path / "results"),
        )
    )

    assert captured["instructions"] == ["Instruction B", "Instruction A"]
    assert captured["completions_A"] == ["no answer", "no answer"]
    assert captured["completions_B"] == ["Answer B", "Answer A"]
    assert prefs.tolist() == [1.0, 1.0]


def test_annotate_battles_warns_when_judge_completions_are_truncated(
    monkeypatch, capsys
):
    captured = {}

    def fake_do_inference(*, chat_model, inputs, use_tqdm):
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
    assert annotations[0].completion_A == "Answer A"
    assert annotations[0].completion_B == "Answer B"
