import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
import yaml

from judgearena.cli import cli
from judgearena.config import ModelArgs, RunConfig
from judgearena.datasets.data import PreferenceData
from judgearena.leaderboard import cli as commands
from judgearena.leaderboard.evaluate import evaluate_release
from judgearena.leaderboard.release import (
    create_release,
    fit_reference_ratings,
    load_release,
)
from judgearena.leaderboard.settings import configured_metrics
from judgearena.leaderboard.submissions import add_submission, submission_table
from judgearena.models import PreparedModel
from judgearena.tasks.registry import load_tasks


def no_inference(*args, **kwargs):
    pytest.fail("cache replay attempted inference")


def metrics(output):
    return json.loads((output / "evaluation.json").read_text())["metrics"]


def fresh_replay(release, output, cache):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from judgearena.config import ModelArgs
from judgearena.models import PreparedModel
from judgearena.leaderboard.evaluate import evaluate_release

def no_inference(*args, **kwargs):
    raise AssertionError('fresh-process cache miss')
PreparedModel.materialize = no_inference
evaluate_release(sys.argv[1], ModelArgs(name='Dummy/candidate', temperature=0.7), sys.argv[2],
                 store_root=sys.argv[3])
""",
            str(release),
            str(output),
            str(cache),
        ],
        check=True,
    )


@pytest.fixture
def arena(tmp_path):
    battles = pd.DataFrame(
        dict(
            battle_id=[f"b{i}" for i in range(6)],
            instruction=[f"Q{i}" for i in range(6)],
            model_a="a",
            model_b="b",
            completion_a="First",
            completion_b="Second",
            reference_pref=[0, 1, 0.5] * 2,
            lang=["en"] * 3 + ["de"] * 3,
            split_group=range(6),
        )
    )
    panel = battles.assign(opponent_slot=["a", "b"] * 3, opponent_model=["a", "b"] * 3)
    cfg = RunConfig(
        task="arena-hard-v0.1",
        model={"name": "a"},
        judge={"model": "Dummy/score A: 6 score B: 8", "swap_mode": "both"},
        protocol={
            "baseline": {"strategy": "none"},
            "judge": {"prompt": {"preset": "default"}, "preference_temperature": 2},
            "scoring": {
                "metrics": [
                    dict(
                        metric="bradley_terry",
                        name=name,
                        parameters=dict(soft=soft, baseline_model="a", n_bootstraps=0),
                    )
                    for name, soft in (("elo", False), ("soft_elo", True))
                ]
                + [dict(metric="pairwise_win_rate")],
            },
        },
        run={"store_root": str(tmp_path / "cache"), "use_tqdm": False},
    )
    return cfg, battles, panel


def test_reference_rating_validation(arena, tmp_path):
    cfg, battles, panel = arena
    values = cfg.model_dump(exclude={"resolved_task"})
    values["protocol"]["judge"]["calibrate_temperature"] = True
    with pytest.raises(ValueError, match="fixed protocol.judge.preference_temperature"):
        create_release(
            RunConfig(**values),
            battles,
            panel,
            tmp_path / "calibrated",
            version="v1",
            labels="judge",
        )
    cfg.judge.model = "Dummy/unparseable"
    release = tmp_path / "human"
    create_release(cfg, battles, panel, release, version="v1", labels="human")
    with pytest.raises(ValueError, match="Cannot submit elo"):
        evaluate_release(
            release, ModelArgs(name="Dummy/candidate"), tmp_path / "invalid"
        )
    assert not (tmp_path / "invalid/release.json").exists()
    configured, _ = configured_metrics(cfg)
    outcomes = battles.assign(pref=battles.reference_pref)
    outcomes.loc[0, "pref"] = float("nan")
    assert len(fit_reference_ratings(outcomes, configured, seed=42)) == 8
    for mask, error in (
        (outcomes.lang == "de", "de.*valid"),
        (outcomes.pref <= 0.5, "connected"),
    ):
        with pytest.raises(ValueError, match=error):
            fit_reference_ratings(
                outcomes.assign(pref=outcomes.pref.mask(mask)), configured, seed=42
            )


def test_frozen_graph_replay_and_collection(tmp_path, arena, monkeypatch):
    cfg, battles, panel = arena
    release, output, collection = [
        tmp_path / p for p in ("release", "first", "entries")
    ]
    cfg.run.seed = 73
    for request in cfg.resolved_task.spec.protocol.scoring.metrics:
        if request.name in ("elo", "soft_elo"):
            request.parameters["n_bootstraps"] = 1 if request.name == "elo" else 2
    create_release(cfg, battles, panel, release, version="v1", labels="judge")

    frozen = (release / "ratings.csv").read_bytes()
    candidate = ModelArgs(name="Dummy/candidate", temperature=0.7)
    evaluate = partial(evaluate_release, release, candidate)
    generated = evaluate(tmp_path / "generated", stage="generate")
    with pytest.raises(FileNotFoundError):
        add_submission(release, generated.parent, collection)
    first = evaluate(output, answers=generated.parent / "answers.parquet")
    receipt = json.loads((output / "release.json").read_text())
    assert receipt["bootstrap_seed"] == 73
    assert receipt["n_bootstraps"] == {"elo": 1, "soft_elo": 2}
    saved = yaml.safe_load((output / "config.yaml").read_text())
    assert saved["model"]["temperature"] == 0.7 and saved["run"]["seed"] == 73
    for name, count in receipt["n_bootstraps"].items():
        assert first.loc[first.method == name, "n_bootstraps"].eq(count).all()
    outcomes = pd.read_parquet(output / "outcomes.parquet")
    assert outcomes.groupby("model_b").completion_b.first().to_dict() == {
        "a": "First",
        "b": "Second",
    }

    def interrupt_write(entry, stream, **kwargs):
        stream.write("{")
        raise InterruptedError

    with monkeypatch.context() as patch:
        patch.setattr(json, "dump", interrupt_write)
        with pytest.raises(InterruptedError):
            add_submission(release, output, collection)
    assert not list(collection.glob("*.json"))
    entry = add_submission(release, output, collection)
    table = submission_table(release, collection, breakdown="lang")
    assert set(table.entry_type) == {"reference", "submission"}
    assert set(table.lang) == {"en", "de"}
    assert set(first.method) == {"elo", "soft_elo", "pairwise_win_rate"}

    cache = tmp_path / "relocated-cache"
    shutil.move(tmp_path / "cache", cache)
    replay = tmp_path / "fresh"
    fresh_replay(release, replay, cache)
    assert metrics(replay) == metrics(output)
    assert add_submission(release, replay, collection) == entry
    assert (release / "ratings.csv").read_bytes() == frozen

    # Even an empty collection must reject a receipt with different interval settings.
    receipt = replay / "release.json"
    receipt.write_text(
        json.dumps({**json.loads(receipt.read_text()), "n_bootstraps": 11})
    )
    with pytest.raises(ValueError, match="Bootstrap settings differ"):
        add_submission(release, replay, tmp_path / "empty-collection")
    receipt.write_text(json.dumps({"version": "other", "files": {}}))
    with pytest.raises(ValueError, match="different release"):
        add_submission(release, replay, collection)
    (output / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        add_submission(release, output, collection)
    (release / "ratings.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_release(release)


def test_disconnected_winrate_release(tmp_path, arena):
    cfg, battles, panel = arena
    candidate = ModelArgs(name="Dummy/candidate")
    # Win-rate metrics do not require a connected rating graph.
    values = cfg.model_dump(exclude={"resolved_task"})
    values["protocol"]["scoring"]["metrics"] = [{"metric": "pairwise_win_rate"}]
    disconnected = pd.concat(
        [
            battles,
            battles.assign(
                battle_id=battles.battle_id + "-other", model_a="c", model_b="d"
            ),
        ]
    )
    winrate = tmp_path / "winrate"
    create_release(RunConfig(**values), disconnected, panel, winrate, version="wr")
    assert evaluate_release(
        winrate, candidate, tmp_path / "wr-run"
    ).method.tolist() == ["pairwise_win_rate"]

    observed = json.loads((winrate / "results.json").read_text())[
        "observed_reference_metrics"
    ]
    assert observed["a"]["pairwise_win_rate"]["winrate"] == 0.5


def test_baseline_cli_replay(tmp_path, monkeypatch):
    data = PreferenceData.from_battles(
        pd.DataFrame(
            {
                "instruction_index": [10, 20],
                "instruction": ["Q1", "Q2"],
                "lang": ["en", "de"],
                "model_a": "b",
                "model_b": "c",
                "completion_a": "Answer",
                "completion_b": "Answer",
            }
        )
    )
    monkeypatch.setattr(commands, "load_task_data", lambda task: data)
    config = dict(
        task="elo-lmarena-140k",
        model={"baseline": ["b", "c"]},
        judge=dict(model="Dummy/[[A>B]]", swap_mode="fixed"),
        protocol=dict(
            judge={"prompt": {"preset": "arena-hard"}},
            scoring={
                "metrics": [{"metric": "pairwise_win_rate", "breakdown_by": ["lang"]}]
            },
        ),
        run=dict(store_root=str(tmp_path / "cache")),
        leaderboard=dict(version="v1"),
    )
    path = tmp_path / "create.yaml"
    create = ["leaderboard", "create", "--config", str(path)]
    path.write_text(yaml.safe_dump(config))
    release, output = tmp_path / "release", tmp_path / "first"
    cli([*create, "--output", str(release)])
    assignments = pd.read_parquet(release / "baselines.parquet")
    assert assignments.groupby("instruction_index").model.apply(set).to_dict() == {
        10: {"b", "c"},
        20: {"b", "c"},
    }
    with pytest.raises(ValueError, match="model.baseline"):
        cli(
            [
                *create,
                "--candidate",
                str(release),
                "--output",
                str(tmp_path / "invalid"),
            ]
        )
    outputs = pd.concat(
        [
            data.model_outputs.assign(completion="Ignore these supplied baselines"),
            data.instructions.reset_index().assign(
                model="candidate", completion="Answer"
            ),
        ]
    )
    evaluate_release(
        release, ModelArgs(name="candidate"), output, model_outputs=outputs
    )
    scores = metrics(output)["pairwise_win_rate"]
    assert scores["winrate"] == 1 and scores["num_wins"] == 4
    assert len(scores["groups"]["lang"]) == 2
    monkeypatch.setattr(PreparedModel, "materialize", no_inference)
    submission = tmp_path / "model.yaml"
    submission.write_text("model: {name: overridden, temperature: 0}\n")
    replay = tmp_path / "replay"
    command = [
        "leaderboard",
        "evaluate",
        "--leaderboard",
        str(release),
        "--config",
        str(submission),
        "--model.name",
        "candidate",
        "--answers",
        str(output / "answers.parquet"),
        "--output",
        str(replay),
    ]
    cli(command)
    assert metrics(replay) == metrics(output)
    for flag, error in (
        ("--model.baseline", "frozen baseline"),
        ("--judge.model", "model, stage settings only"),
    ):
        with pytest.raises(ValueError, match=error):
            cli(command + [flag, "changed"])
    for flag in ("--seed", "--n-bootstraps"):
        with pytest.raises(SystemExit):
            cli(command + [flag, "7"])


def test_custom_criteria_task_and_portable_replay(tmp_path, capsys):
    config = yaml.safe_load(
        Path("configs/leaderboard/arena-hard-quality-v1.yaml").read_text()
    )
    rubric = tmp_path / "rubric.yaml"
    shutil.copyfile("configs/leaderboard/quality-rubric.yaml", rubric)
    config["judge"]["model"] = 'Dummy/{"criteria": {"clarity": 8, "completeness": 6}}'
    config["run"]["store_root"] = str(tmp_path / "cache")
    components = tmp_path / "tasks/custom/components.py"
    components.parent.mkdir(parents=True)
    components.write_text("""import pandas as pd
from judgearena.datasets.registry import DatasetAdapter
from judgearena.datasets.data import CompletionData
class Adapter(DatasetAdapter):
    def load_data(self):
        return CompletionData(pd.DataFrame({'instruction':['Q1','Q2'], 'lang':['en','de']}))
class Metric:
    def calculate(self, rows):
        return {'custom': {'count': len(rows)}}
    def render(self, values):
        return str(values)
""")
    config["protocol"]["judge"]["prompt"]["options"]["criteria_file"] = str(rubric)
    cfg = RunConfig(
        model={"name": "Dummy/candidate"},
        **{k: v for k, v in config.items() if k != "leaderboard"},
    )
    task = cfg.resolved_task.spec.model_dump(mode="json")
    task.update(task="custom-quality", dataset={"adapter": "./components.py:Adapter"})
    task["protocol"]["scoring"]["metrics"].append(
        dict(metric="./components.py:Metric", name="custom", breakdown_by=["lang"])
    )
    (components.parent / "custom.yaml").write_text(yaml.safe_dump(task))
    cfg = RunConfig(
        resolved_task=load_tasks(components.parent.parent)["custom-quality"],
        model=cfg.model,
        judge=cfg.judge,
        run=cfg.run,
    )
    config = {**cfg.model_dump(mode="json"), "leaderboard": {"version": "custom"}}
    path, release = tmp_path / "config.yaml", tmp_path / "release"
    path.write_text(yaml.safe_dump(config))
    create_command = ["leaderboard", "create", "--config", str(path)]
    cli([*create_command, "--output", str(release)])
    with pytest.raises(ValueError, match="generation-only"):
        cli(
            [
                *create_command,
                "--stage",
                "generate",
                "--output",
                str(tmp_path / "invalid-stage"),
            ]
        )
    rubric.unlink()
    shutil.rmtree(components.parent)
    for name in ("candidate", "second"):
        output = tmp_path / name
        evaluate_release(
            release, ModelArgs(name=f"Dummy/{name}", temperature=0.7), output
        )
    add = partial(add_submission, release, collection=tmp_path / "entries")
    with ThreadPoolExecutor(max_workers=3) as pool:
        entries = list(
            pool.map(
                add, [tmp_path / name for name in ("candidate", "second", "candidate")]
            )
        )
    assert entries[0] == entries[2] != entries[1]
    table = submission_table(release, tmp_path / "entries")
    assert set(table["score"].dropna()) == {7}
    assert set(table["criteria.clarity.score"].dropna()) == {8}
    assert set(table["criteria.completeness.score"].dropna()) == {6}
    overall = submission_table(release, tmp_path / "entries", overall=True)
    assert overall["custom.count"].dropna().tolist() == [2, 2]
    with pytest.raises(ValueError, match="Unknown breakdown"):
        submission_table(release, tmp_path / "entries", breakdown="missing")
    capsys.readouterr()
    cli(
        [
            "leaderboard",
            "table",
            "--leaderboard",
            str(release),
            "--collection",
            str(tmp_path / "entries"),
            "--breakdown",
            "lang",
            "--columns",
            "model",
            "custom.count",
        ]
    )
    displayed = capsys.readouterr().out
    rows = [line.split() for line in displayed.splitlines()]
    assert rows[0] == ["lang", "model", "custom.count"]
    assert sorted(rows[1:]) == sorted(
        [lang, f"Dummy/{model}", "1"]
        for lang in ("en", "de")
        for model in ("candidate", "second")
    )

    entry_path = next((tmp_path / "entries").glob("*.json"))
    entry = json.loads(entry_path.read_text())
    entry["result"]["metrics"]["custom"]["groups"]["category"] = []
    entry_path.write_text(json.dumps(entry))
    empty = submission_table(
        release,
        tmp_path / "entries",
        breakdown="category",
        columns=["model", "custom.count"],
    )
    assert empty.empty and list(empty.columns) == ["category", "model", "custom.count"]

    fresh_replay(release, tmp_path / "fresh", tmp_path / "cache")
    assert metrics(tmp_path / "candidate") == metrics(tmp_path / "fresh")
