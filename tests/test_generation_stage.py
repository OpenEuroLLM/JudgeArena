import json
import shutil
import subprocess
import sys
from dataclasses import replace

import pandas as pd
import pytest

from judgearena.benchmarks.runner import run_benchmark
from judgearena.cli import cli
from judgearena.config import RunConfig, load_config
from judgearena.datasets.data import CompletionData
from judgearena.models import PreparedModel


@pytest.mark.parametrize("runner", ["pairwise", "single_answer"])
def test_generate_replay_then_judge_saved_answers(tmp_path, monkeypatch, runner):
    protocol = dict(
        runner=runner,
        generation={"mode": "provided_messages"},
        judge={
            "prompt": {
                "preset": "single-score" if runner == "single_answer" else "default"
            }
        },
        scoring={
            "metrics": [
                {
                    "metric": "mean_score"
                    if runner == "single_answer"
                    else "pairwise_win_rate"
                }
            ]
        },
    )
    if runner == "pairwise":
        protocol["baseline"] = {"strategy": "runtime_required"}
    cfg = RunConfig(
        stage="generate",
        model={"name": "Dummy/answer"},
        protocol=protocol,
        dataset={"adapter": "arena_hard"},
        run={"store_root": str(tmp_path / "cache"), "use_tqdm": False},
    )
    instructions = pd.DataFrame(
        {
            "instruction": ["Q"],
            "messages": [[{"role": "user", "content": "Q"}]],
            "lang": ["en"],
        },
        index=pd.Index(["q"], name="instruction_index"),
    )
    data = CompletionData(instructions)
    monkeypatch.setattr(
        "judgearena.benchmarks.evaluation.build_judge",
        lambda *a, **k: pytest.fail("generation constructed judge"),
    )
    generated = run_benchmark(cfg, data=data, output_dir=tmp_path / "generated")
    assert generated.name == "generation.json"
    assert not (generated.parent / "evaluation.json").exists()
    answers = pd.read_parquet(generated.parent / "answers.parquet")
    assert answers.completion.tolist() == ["answer"]
    assert len(answers.messages.iloc[0]) == 2
    with monkeypatch.context() as block:
        block.setattr(
            PreparedModel, "materialize", lambda *a: pytest.fail("cache miss")
        )
        run_benchmark(
            load_config(generated.parent / "config.yaml"),
            data=data,
            output_dir=tmp_path / "replay",
        )
        block.setattr(
            "judgearena.benchmarks.evaluation.load_task_data", lambda task: data
        )
        cli(
            [
                "--config",
                str(generated.parent / "config.yaml"),
                "--stage",
                "generate",
                "--run.result_folder",
                str(tmp_path / "cli-replay"),
            ]
        )
    if runner == "single_answer":
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
import pandas as pd
from judgearena.config import load_config
from judgearena.datasets.data import CompletionData, restore_nested_columns
from judgearena.benchmarks.runner import run_benchmark
from judgearena.models import PreparedModel

def blocked(*args, **kwargs):
    raise AssertionError('fresh-process generation cache miss')
PreparedModel.materialize = blocked
cfg = load_config(sys.argv[1] + '/config.yaml')
data = CompletionData(restore_nested_columns(pd.read_parquet(sys.argv[1] + '/instructions.parquet')))
run_benchmark(cfg, data=data, output_dir=sys.argv[2])
""",
                str(generated.parent),
                str(tmp_path / "fresh"),
            ],
            check=True,
        )
    monkeypatch.undo()
    values = cfg.model_dump()
    values.update(
        stage="evaluate",
        answers=str(generated.parent / "answers.parquet"),
        judge={
            "model": 'Dummy/{"score": 7}'
            if runner == "single_answer"
            else "Dummy/score A: 6 score B: 8"
        },
    )
    if runner == "pairwise":
        values["model"]["baseline"] = "baseline"
        data = replace(
            data,
            model_outputs=pd.DataFrame(
                [
                    {
                        "instruction_index": "q",
                        "model": "baseline",
                        "completion": "reference",
                        "messages": [
                            {"role": "user", "content": "Q"},
                            {"role": "assistant", "content": "reference"},
                        ],
                    }
                ]
            ),
        )
    judged = RunConfig(**values)
    monkeypatch.setattr(
        "judgearena.benchmarks.execution.build_generation_kwargs",
        lambda *a, **k: pytest.fail("regenerated saved answer"),
    )
    result = run_benchmark(judged, data=data, output_dir=tmp_path / "judged")
    assert json.loads(result.read_text())["metrics"]
    with pytest.raises(ValueError, match="instructions"):
        run_benchmark(
            judged,
            data=replace(
                data, instructions=instructions.assign(instruction="different")
            ),
            output_dir=tmp_path / "wrong",
        )

    shutil.rmtree(generated.parent)
    with monkeypatch.context() as block:
        block.setattr(
            PreparedModel,
            "materialize",
            lambda *a: pytest.fail("evaluation cache miss"),
        )
        replayed = run_benchmark(
            load_config(result.parent / "config.yaml"),
            data=data,
            output_dir=tmp_path / "judged-replay",
        )
    assert (
        json.loads(replayed.read_text())["metrics"]
        == json.loads(result.read_text())["metrics"]
    )
