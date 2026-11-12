import json
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
import yaml

from judgearena.cli import cli
from judgearena.config import ModelArgs, RunConfig
from judgearena.datasets.data import CompletionData, PreferenceData
from judgearena.leaderboard import cli as commands
from judgearena.leaderboard import release as releases
from judgearena.leaderboard.evaluate import evaluate_release
from judgearena.leaderboard.release import create_release, load_release
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
evaluate_release(sys.argv[1], ModelArgs(name='Dummy/candidate'), sys.argv[2],
                 store_root=sys.argv[3], n_bootstraps=10)
""",
            str(release),
            str(output),
            str(cache),
        ],
        check=True,
    )


@pytest.mark.parametrize("runner", ["elo", "pairwise"])
def test_frozen_graph_replay_and_collection(tmp_path, monkeypatch, runner, capsys):
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
        task="elo-lmarena-140k",
        model={"name": "a"},
        judge={"model": "Dummy/score A: 6 score B: 8", "swap_mode": "both"},
        protocol={
            "scoring": {
                "default_temperature": 2,
                "metrics": [
                    dict(metric="bradley_terry", name=name, parameters=dict(soft=soft))
                    for name, soft in (("elo", False), ("soft_elo", True))
                ]
                + [dict(metric="pairwise_win_rate")],
            }
        },
        elo={"baseline_model": "a", "n_bootstraps": 0},
        run={"store_root": str(tmp_path / "cache"), "use_tqdm": False},
    )
    if runner == "pairwise":
        protocol = cfg.resolved_task.spec.protocol.model_dump(mode="json")
        protocol.pop("arena")
        protocol["runner"] = "pairwise"
        protocol["scoring"].pop("default_soft")
        protocol["scoring"].pop("default_temperature")
        protocol["judge"]["preference_temperature"] = 2
        for request in protocol["scoring"]["metrics"]:
            if request["metric"] == "bradley_terry":
                request["parameters"].update(baseline_model="a", n_bootstraps=0)
        cfg = RunConfig(
            task="arena-hard-v0.1",
            model=cfg.model,
            judge=cfg.judge,
            protocol=protocol,
            run=cfg.run,
        )
    release, output, collection = [
        tmp_path / p for p in ("release", "first", "entries")
    ]
    create = partial(create_release, cfg, battles, panel, version="v1", labels="judge")
    if cfg.elo is not None:
        cfg.elo.calibrate_temperature = True
        from judgearena.benchmarks.pairwise.judging import judge_configured_comparisons
        from judgearena.prompts.parsing import score_preference

        _, raw, _ = judge_configured_comparisons(cfg, battles)
        assert raw.pref.iloc[0] == pytest.approx(
            score_preference(6, 8, temperature=0.3)
        )
        with pytest.raises(ValueError, match="fixed soft_elo_temperature"):
            create(release)
        assert not release.exists()
        cfg.elo.calibrate_temperature = False
    create(release)
    from judgearena.prompts.parsing import score_preference

    assert pd.read_parquet(release / "outcomes.parquet").pref.iloc[0] == pytest.approx(
        score_preference(6, 8, temperature=2)
    )
    frozen = (release / "ratings.csv").read_bytes()
    candidate = ModelArgs(name="Dummy/candidate")
    evaluate = partial(evaluate_release, release, candidate, n_bootstraps=10)
    first = evaluate(output)
    assert "lang=en:" in capsys.readouterr().out
    outcomes = pd.read_parquet(output / "outcomes.parquet")
    assert outcomes.groupby("model_b").completion_b.first().to_dict() == {
        "a": "First",
        "b": "Second",
    }
    add_submission(release, output, collection)
    monkeypatch.setattr(PreparedModel, "materialize", no_inference)
    from judgearena.benchmarks import evaluation as execution

    panel_judge = execution.judge_configured_comparisons

    def missing_preferences(*args, **kwargs):
        annotations, outcomes, metadata = panel_judge(*args, **kwargs)
        return annotations, outcomes.assign(pref=float("nan")), metadata

    with monkeypatch.context() as patch:
        patch.setattr(execution, "judge_configured_comparisons", missing_preferences)
        with pytest.raises(ValueError, match="Cannot submit elo"):
            evaluate(tmp_path / "missing-judgments")
    assert not (tmp_path / "missing-judgments/release.json").exists()
    replay = tmp_path / "replay"
    pd.testing.assert_frame_equal(first, evaluate(replay))
    assert add_submission(release, replay, collection) == next(
        collection.glob("*.json")
    )
    assert len(list(collection.glob("*.json"))) == 1
    table = submission_table(release, collection)
    references = table[table.entry_type == "reference"]
    assert len(references) == 8 and len(table) == 13
    assert set(first.method) == {"elo", "soft_elo", "pairwise_win_rate"}
    assert set(first.lang.dropna()) == {"en", "de"}
    evaluate(tmp_path / "conflict", n_bootstraps=11)
    with pytest.raises(ValueError, match="Conflicting"):
        add_submission(release, tmp_path / "conflict", collection)
    result_path = output / "results.json"
    original = result_path.read_text()
    result_path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        add_submission(release, output, collection)
    result_path.write_text(original)
    receipt = replay / "release.json"
    receipt.write_text(json.dumps({"version": "other", "files": {}}))
    with pytest.raises(ValueError, match="different release"):
        add_submission(release, replay, collection)

    # Missing judgments may be excluded, but cannot remove a model or disconnect Elo.
    original_judge = releases.judge_comparisons
    for missing, error in (
        ("one", None),
        ("language", "de.*valid"),
        ("wins", "connected"),
    ):

        def judge(*args, missing=missing, **kwargs):
            annotations, outcomes, metadata = original_judge(*args, **kwargs)
            mask = {
                "one": outcomes.index == outcomes.index[0],
                "language": outcomes.lang == "de",
                "wins": outcomes.pref < 0.5,
            }[missing]
            return annotations, outcomes.assign(pref=outcomes.pref.mask(mask)), metadata

        monkeypatch.setattr(releases, "judge_comparisons", judge)
        destination = tmp_path / missing
        if error:
            with pytest.raises(ValueError, match=error):
                create(destination)
            assert not destination.exists()
        else:
            create(destination)
            assert len(load_release(destination)[1]) == 8
            assert (
                pd.read_parquet(destination / "outcomes.parquet").pref.isna().sum() == 1
            )

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
    with pytest.raises(ValueError, match="Collection contains a different release"):
        add_submission(winrate, tmp_path / "wr-run", collection)

    cache = tmp_path / "relocated-cache"
    shutil.move(tmp_path / "cache", cache)
    fresh_replay(release, tmp_path / "fresh", cache)
    assert metrics(output) == metrics(tmp_path / "fresh")
    add_submission(release, tmp_path / "fresh", collection)
    assert (release / "ratings.csv").read_bytes() == frozen
    (release / "ratings.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_release(release)


@pytest.mark.parametrize("with_comparisons", [False, True])
def test_baseline_cli_replay(tmp_path, monkeypatch, with_comparisons):
    instructions = pd.DataFrame(
        dict(
            instruction=["Q1", "Q2"],
            lang=["en", "de"],
            category=["coding", "writing"],
            messages=[[{"role": "user", "content": f"Q{i}"}] for i in (1, 2)],
        ),
        index=pd.Index([10, 20], name="instruction_index"),
    )
    outputs = pd.DataFrame(
        [
            dict(
                instruction_index=i,
                model=model,
                completion="Answer",
                detail="kept",
                messages=row.messages + [{"role": "assistant", "content": "Answer"}],
            )
            for model in ("b", "c", "candidate")
            for i, row in instructions.iterrows()
        ]
    )
    data = CompletionData(instructions, outputs[outputs.model != "candidate"])
    if with_comparisons:
        data = PreferenceData(
            data.instructions,
            data.model_outputs,
            battles=instructions.reset_index().assign(
                model_a="b", model_b="c", completion_a="Answer", completion_b="Answer"
            ),
        )
    monkeypatch.setattr(commands, "load_task_data", lambda task: data)
    config = dict(
        task="arena-hard-v0.1",
        judge=dict(model="Dummy/[[A>B]]", swap_mode="fixed"),
        protocol=dict(
            generation=dict(mode="provided_messages"),
            baseline=dict(strategy="all_references", references=["b", "c"]),
            judge=dict(turns="last", category_prompts={"coding": "arena-hard"}),
            scoring=dict(
                metrics=[dict(metric="pairwise_win_rate", breakdown_by=["lang"])]
            ),
        ),
        run=dict(store_root=str(tmp_path / "cache")),
        leaderboard=dict(version="v1"),
    )
    path, release, output = [tmp_path / p for p in ("create.yaml", "release", "first")]
    path.write_text(yaml.safe_dump(config))
    cli(["leaderboard", "create", "--config", str(path), "--output", str(release)])
    with pytest.raises(ValueError, match="baseline.strategy: none"):
        cli(
            [
                "leaderboard",
                "create",
                "--config",
                str(path),
                "--candidate",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "invalid"),
            ]
        )
    assert set(pd.read_parquet(release / "model_outputs.parquet").detail) == {"kept"}
    outputs.loc[outputs.model != "candidate", "completion"] = (
        "Ignore these supplied baselines"
    )
    evaluate_release(
        release, ModelArgs(name="candidate"), output, model_outputs=outputs
    )
    scores = metrics(output)["pairwise_win_rate"]
    assert scores["winrate"] == 1 and scores["num_wins"] == 4
    assert len(scores["groups"]["lang"]) == 2
    collection = tmp_path / "entries"
    add_submission(release, output, collection)
    monkeypatch.setattr(PreparedModel, "materialize", no_inference)
    submission = tmp_path / "model.yaml"
    submission.write_text("model: {name: overridden, temperature: 0}\n")
    for index, extra in enumerate(([], ["--config", str(submission)])):
        replay = tmp_path / f"replay-{index}"
        command = [
            "leaderboard",
            "evaluate",
            "--leaderboard",
            str(release),
            "--answers",
            str(output / "answers.parquet"),
            "--output",
            str(replay),
            *extra,
            "--model.name",
            "candidate",
            "--n-bootstraps",
            "11",
        ]
        cli(command)
        assert metrics(replay) == metrics(output)
        add_submission(release, replay, collection)
    table = submission_table(release, collection)
    assert len(table) == 3 and set(table.winrate) == {1}
    for flag, error in (
        ("--model.baseline", "frozen baseline"),
        ("--judge.model", "model settings only"),
    ):
        with pytest.raises(ValueError, match=error):
            cli(command + [flag, "changed"])


def test_custom_criteria_task_and_portable_replay(tmp_path, monkeypatch):
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
from judgearena.datasets.data import CompletionData, PreferenceData
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
    cli(["leaderboard", "create", "--config", str(path), "--output", str(release)])
    bad_prompt = dict(
        judging_mode="single_answer", parser="single-score", user="{missing_context}"
    )
    with pytest.raises(ValueError, match="missing_context"):
        cli(
            [
                "leaderboard",
                "create",
                "--config",
                str(path),
                "--judge.prompt",
                json.dumps(bad_prompt),
                "--output",
                str(tmp_path / "invalid"),
            ]
        )
    assert not (tmp_path / "invalid").exists()
    rubric.unlink()
    shutil.rmtree(components.parent)
    for name in ("candidate", "second"):
        output = tmp_path / name
        evaluate_release(release, ModelArgs(name=f"Dummy/{name}"), output)
        add_submission(release, output, tmp_path / "entries")
    table = submission_table(release, tmp_path / "entries")
    assert set(table["score"].dropna()) == {7}
    assert set(table["criteria.clarity.score"].dropna()) == {8}
    assert set(table["criteria.completeness.score"].dropna()) == {6}
    selected = submission_table(
        release,
        tmp_path / "entries",
        columns=["model", "custom.count"],
        sort_by="custom.count",
        ascending=False,
    )
    assert selected["custom.count"].dropna().tolist() == [2, 2, 1, 1, 1, 1]
    monkeypatch.setattr(PreparedModel, "materialize", no_inference)
    evaluate_release(
        release,
        ModelArgs(name="Dummy/candidate"),
        tmp_path / "replay",
        seed=99,
        n_bootstraps=11,
    )
    assert metrics(tmp_path / "candidate") == metrics(tmp_path / "replay")
    add_submission(release, tmp_path / "replay", tmp_path / "entries")
    fresh_replay(release, tmp_path / "fresh", tmp_path / "cache")
    assert metrics(tmp_path / "candidate") == metrics(tmp_path / "fresh")
