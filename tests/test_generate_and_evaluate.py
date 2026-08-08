import pandas as pd
import pytest

import judgearena.generate_and_evaluate as generate_and_evaluate
import judgearena.utils as utils
from judgearena.generate_and_evaluate import (
    CliArgs,
)
from judgearena.generate_and_evaluate import (
    main as main_generate_and_eval,
)


@pytest.fixture(autouse=True)
def mock_external_data_and_cache(monkeypatch):
    instructions = pd.DataFrame(
        {
            "instruction": [f"Synthetic instruction {i}" for i in range(20)],
        },
        index=pd.Index(range(20), name="instruction_index"),
    )

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
        "load_contexts",
        lambda dataset: instructions.loc[:, "instruction"],
    )

    monkeypatch.setattr(
        generate_and_evaluate,
        "try_load_dataset_completions",
        lambda dataset, model, n_instructions: None,
    )


@pytest.mark.parametrize(
    "task",
    [
        "alpaca-eval",
        "arena-hard-v2.0",
        "arena-hard-v0.1",
        "fluency-french",
        "m-arena-hard-EU",
    ],
)
def test_generate_and_evaluate_context_completion(task: str, tmp_path):
    prefs = main_generate_and_eval(
        CliArgs(
            task=task,
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
            result_folder=str(tmp_path),
            # default for swap_mode is "fixed"
        )
    )

    avg_pref = sum(prefs) / len(prefs)
    assert avg_pref >= 0.9


def test_generate_and_evaluate_correct_order_bias(tmp_path):
    """Test the correction for model order bias.

    In this test, a judge that is totally biased towards model B should be corrected to be neutral.
    Since the judge favors model B regardless of the order and the completions, the average
    preference should be 0.5.
    """
    prefs = main_generate_and_eval(
        CliArgs(
            task="alpaca-eval",
            model_A="Dummy/no answer",
            model_B="Dummy/open is better than close isnt'it",
            judge_model="Dummy/score A: 0 score B: 10",
            n_instructions=5,
            swap_mode="both",
            result_folder=str(tmp_path),
        )
    )

    avg_pref = sum(prefs) / len(prefs)
    assert avg_pref == 0.5


def test_generate_and_evaluate_reuses_inference_cache(tmp_path, monkeypatch):
    args = CliArgs(
        task="alpaca-eval",
        model_A="Dummy/answer A",
        model_B="Dummy/answer B",
        judge_model="Dummy/score A: 1 score B: 0",
        n_instructions=2,
        result_folder=str(tmp_path / "results"),
        store_root=str(tmp_path / "cache"),
    )
    main_generate_and_eval(args)
    monkeypatch.setattr(
        utils,
        "make_model",
        lambda *_args, **_kwargs: pytest.fail("cache hit materialized a model"),
    )
    main_generate_and_eval(args)
