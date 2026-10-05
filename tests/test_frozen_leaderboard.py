"""Focused tests for the frozen-anchor leaderboard boundary."""

import json
from pathlib import Path

import pandas as pd
import pytest

import judgearena.benchmarks.elo.runner as elo_runner
import judgearena.models as models
from judgearena.benchmarks.elo.artifacts import BATTLE_COLUMNS
from judgearena.benchmarks.elo.leaderboard import (
    AnchorSet,
    LeaderboardEntry,
    build_leaderboard,
    collapse_swapped_rows,
    comparable_config,
    protocol_identifier,
    rebuild_leaderboard,
    score_frozen_submission,
    write_entry,
)
from judgearena.benchmarks.elo.runner import run_elo
from judgearena.config import RunConfig, dump_config, load_config
from judgearena.tasks.registry import get_packaged_task


def _anchors():
    return AnchorSet(
        name="euro-test",
        task="elo-test",
        arena="test-arena",
        baseline_model="reference",
        protocol_id="frozen-protocol",
        languages=("en", "fr"),
        ratings_by_language={
            "en": {"reference": 1000.0, "strong": 1200.0},
            "fr": {"reference": 1000.0, "strong": 1100.0},
        },
        counts_by_language={
            "en": {"reference": 20, "strong": 10},
            "fr": {"reference": 30, "strong": 15},
        },
        human_battles_by_language={"en": 40, "fr": 50},
        battles_per_language=4,
        bootstrap_seed=19,
    )


def _battles(*, duplicate=False):
    rows = []
    for language, prefs in (("en", [0.0, 0.0, 0.0, 1.0]), ("fr", [0.0, 1.0, 1.0, 1.0])):
        for index, pref in enumerate(prefs):
            row = {
                "panel_id": f"{language}-{index}",
                "lang": language,
                "model_a": "candidate",
                "model_b": "reference",
                "pref": pref,
            }
            if duplicate:
                rows.extend(
                    [row | {"orientation": "direct"}, row | {"orientation": "reversed"}]
                )
            else:
                rows.append(row)
    return pd.DataFrame(rows)


def test_anchor_round_trip(tmp_path):
    anchors = _anchors()
    path = anchors.save(tmp_path / "anchors.json")

    assert AnchorSet.load(path) == anchors
    assert anchors.overall_ratings == {"reference": 1000.0, "strong": 1150.0}


def test_protocol_identity_covers_anchor_material_and_ordered_panel():
    panel = pd.DataFrame([{"panel_id": "one"}, {"panel_id": "two"}])
    material = _anchors()
    config = {"task": "elo-test"}
    identity = protocol_identifier(material, panel, config)

    assert (
        identity == "67ccee9c3a6cab65a7c5ed64696f06ea74a772ec4db2a038735467a67f2dbe81"
    )
    changed = material.model_copy(update={"bootstrap_seed": 20})
    assert protocol_identifier(changed, panel, config) != identity
    assert protocol_identifier(material, panel, {"task": "other"}) != identity
    assert protocol_identifier(material, panel.iloc[::-1], config) != identity
    with pytest.raises(ValueError, match="Out of range float"):
        protocol_identifier(material, panel.assign(pref=float("nan")), config)


def test_swap_rows_are_required_and_averaged_by_panel_id():
    rows = pd.DataFrame(
        [
            {
                "panel_id": "p",
                "lang": "en",
                "model_a": "candidate",
                "model_b": "reference",
                "pref": 0.2,
                "orientation": "direct",
            },
            {
                "panel_id": "p",
                "lang": "en",
                "model_a": "candidate",
                "model_b": "reference",
                "pref": 0.6,
                "orientation": "reversed",
            },
        ]
    )
    collapsed = collapse_swapped_rows(rows, "both")

    assert collapsed.to_dict("records") == [
        {
            "panel_id": "p",
            "lang": "en",
            "model_a": "candidate",
            "model_b": "reference",
            "pref": 0.4,
        }
    ]
    with pytest.raises(ValueError, match="expected 2"):
        collapse_swapped_rows(rows.iloc[:1], "both")
    hard_rows = rows.assign(pref_hard=[0.0, 1.0])
    assert collapse_swapped_rows(hard_rows, "both").loc[0, "pref_hard"] == 0.5
    duplicated = rows.assign(orientation="direct")
    with pytest.raises(ValueError, match="invalid orientations"):
        collapse_swapped_rows(duplicated, "both")
    with pytest.raises(ValueError, match="between zero and one"):
        collapse_swapped_rows(rows.assign(pref=[-1.0, 2.0]), "both")


def _score(
    battles, candidate="candidate", anchors=None, *, soft_elo=True, n_bootstraps=20
):
    return score_frozen_submission(
        battles,
        candidate,
        anchors or _anchors(),
        soft_elo=soft_elo,
        n_bootstraps=n_bootstraps,
    )


def test_multilingual_score_uses_point_estimates_weights_and_bootstrap_units():
    entry = _score(_battles())

    assert entry.by_language["en"].rating == pytest.approx(1000 + 400 * 0.4771212547)
    assert entry.by_language["fr"].rating == pytest.approx(1000 - 400 * 0.4771212547)
    expected = (entry.by_language["en"].rating + entry.by_language["fr"].rating) / 2
    assert entry.overall.rating == pytest.approx(expected)
    assert entry.overall.n_battles == 8
    assert entry.overall.ci_low is not None
    assert entry.by_language["en"].ci_high is not None


def test_hard_frozen_score_uses_quantized_preferences():
    battles = _battles().assign(pref=0.5, pref_hard=0.0)

    entry = _score(battles, soft_elo=False, n_bootstraps=0)

    assert entry.overall.rating == 2000.0


def test_swap_scoring_counts_prompts_not_judge_passes():
    battles = collapse_swapped_rows(_battles(duplicate=True), "both")
    entry = _score(battles, n_bootstraps=0)

    assert entry.overall.n_battles == 8
    assert entry.overall.ci_low is None
    assert entry.by_language["en"].rating > entry.by_language["fr"].rating


def test_scoring_requires_complete_languages_and_known_opponents():
    battles = _battles()
    with pytest.raises(ValueError, match="languages must be exactly"):
        _score(battles[battles.lang == "en"])

    battles.loc[0, "model_b"] = "unknown"
    with pytest.raises(ValueError, match="Unknown frozen opponents"):
        _score(battles)


def test_exclusive_entry_write_and_atomic_rebuild(tmp_path):
    anchors = _anchors()
    entry = _score(_battles(), anchors=anchors, n_bootstraps=0)
    entry_path = write_entry(tmp_path, entry)
    with pytest.raises(FileExistsError):
        write_entry(tmp_path, entry)

    board_path = rebuild_leaderboard(tmp_path, anchors)
    board = json.loads(board_path.read_text())
    assert board["protocol_id"] == anchors.protocol_id
    assert {row["source"] for row in board["entries"]} == {"anchor", "submission"}
    ratings = [row["overall"]["rating"] for row in board["entries"]]
    assert ratings == sorted(ratings, reverse=True)

    old_board = board_path.read_text()
    invalid = LeaderboardEntry.model_validate_json(entry_path.read_text()).model_copy(
        update={"protocol_id": "other"}
    )
    entry_path.write_text(invalid.model_dump_json(by_alias=True))
    with pytest.raises(ValueError, match="protocol mismatch"):
        rebuild_leaderboard(tmp_path, anchors)
    assert board_path.read_text() == old_board


def test_build_rejects_duplicate_models():
    anchors = _anchors()
    battles = _battles().replace({"candidate": "strong"})
    entry = _score(battles, "strong", anchors, n_bootstraps=0)
    with pytest.raises(ValueError, match="duplicate leaderboard model"):
        build_leaderboard(anchors, [entry])
    invalid = entry.model_copy(
        update={
            "model": "other",
            "overall": entry.overall.model_copy(update={"rating": 0}),
        }
    )
    with pytest.raises(ValueError, match="incorrect overall rating"):
        build_leaderboard(anchors, [invalid])


@pytest.mark.parametrize("swap_mode", ["fixed", "both"])
def test_unparseable_comparison_is_skipped_and_partial_entry_is_valid(
    swap_mode, tmp_path
):
    battles = _battles(duplicate=swap_mode == "both")
    failed_row = 1 if swap_mode == "both" else 0
    battles.loc[failed_row, "pref"] = float("nan")
    complete = collapse_swapped_rows(battles, swap_mode)

    assert "en-0" not in set(complete.panel_id)
    assert len(complete) == 7
    entry = _score(complete, n_bootstraps=5)
    assert entry.overall.n_battles == 7
    assert entry.by_language["en"].n_battles == 3
    assert entry.by_language["fr"].n_battles == 4
    write_entry(tmp_path, entry)
    board = json.loads(rebuild_leaderboard(tmp_path, _anchors()).read_text())
    candidate = next(row for row in board["entries"] if row["source"] == "submission")
    assert candidate["overall"]["n_battles"] == 7


def test_no_valid_judgments_for_a_language_cannot_produce_a_rating():
    battles = _battles(duplicate=True)
    battles.loc[battles.lang.eq("en"), "pref"] = float("nan")
    complete = collapse_swapped_rows(battles, "both")
    with pytest.raises(ValueError, match="languages must be exactly"):
        _score(complete)


def _frozen_run_config(tmp_path, leaderboard_dir, model: str) -> RunConfig:
    return RunConfig(
        task="elo-comparia",
        model={"name": model},
        judge={"model": "Dummy/score A: 0 score B: 10", "swap_mode": "fixed"},
        generation={"n_instructions": None},
        elo={
            "baseline_model": "anchor",
            "languages": ["en", "fr"],
            "n_bootstraps": 3,
            "calibrate_temperature": False,
            "leaderboard_dir": leaderboard_dir,
        },
        run={"result_folder": str(tmp_path / "results"), "store_root": None},
    )


def _write_frozen_run_files(cfg: RunConfig, directory, task) -> None:
    assert cfg.elo is not None
    cfg.elo = cfg.elo.resolve(task.spec.protocol.scoring)
    panel = pd.DataFrame(
        [
            {
                "panel_id": f"{language}-{index}",
                "question_id": f"q-{language}-{index}",
                "lang": language,
                "instruction": f"Instruction {language} {index}",
                "opponent_model": "anchor",
                "opponent_completion": f"Anchor response {language} {index}",
                "candidate_position": "A" if index % 2 == 0 else "B",
            }
            for language in ("en", "fr")
            for index in range(2)
        ]
    )
    data = {
        "name": "test-leaderboard",
        "version": "0.01",
        "min_anchor_battles": 1,
        "dataset_sources": {},
        "task": task.task,
        "arena": task.spec.protocol.arena,
        "baseline_model": "anchor",
        "languages": ("en", "fr"),
        "ratings_by_language": {
            "en": {"anchor": 1000.0},
            "fr": {"anchor": 1000.0},
        },
        "counts_by_language": {"en": {"anchor": 20}, "fr": {"anchor": 20}},
        "human_battles_by_language": {"en": 30, "fr": 30},
        "battles_per_language": 2,
        "bootstrap_seed": 0,
        "schema_version": 1,
    }
    directory.mkdir()
    frozen_cfg = cfg.model_copy(
        update={"elo": cfg.elo.model_copy(update={"leaderboard_dir": None})}
    )
    config_path = directory / "config.yaml"
    dump_config(frozen_cfg, config_path)
    frozen_config = load_config(config_path)
    anchors = AnchorSet(**data, protocol_id="")
    anchors = anchors.model_copy(
        update={
            "protocol_id": protocol_identifier(
                anchors, panel, comparable_config(frozen_config)
            )
        }
    )
    panel.to_parquet(directory / "panel.parquet", index=False)
    anchors.save(directory / "anchors.json")
    (directory / "entries").mkdir()
    rebuild_leaderboard(directory, anchors)


def test_frozen_runner_adds_independent_immutable_entries(monkeypatch, tmp_path):
    directory = tmp_path / "leaderboard"
    task = get_packaged_task("elo-comparia")
    cfg_a = _frozen_run_config(tmp_path, directory, "Dummy/a/b")
    _write_frozen_run_files(cfg_a, directory, task)
    monkeypatch.setattr(
        elo_runner,
        "load_battles",
        lambda _task: pytest.fail("frozen runs must not reload arena battles"),
    )
    first = run_elo(cfg_a, task)
    board_after_a = json.loads((directory / "leaderboard.json").read_text())
    entry_a = next(
        row for row in board_after_a["entries"] if row["model"] == "Dummy/a/b"
    )
    with pytest.raises(ValueError, match="already on the leaderboard"):
        run_elo(_frozen_run_config(tmp_path, directory, "Dummy/a/b"), task)

    cfg_b = _frozen_run_config(tmp_path, directory, "Dummy/a_b")
    second = run_elo(cfg_b, task)
    board_after_b = json.loads((directory / "leaderboard.json").read_text())

    assert board_after_a["protocol_id"] in first["result_path"]
    assert first["result_path"] != second["result_path"]
    assert Path(first["result_path"]).is_file()
    assert Path(second["result_path"]).is_file()
    assert first["sampling_metadata"]["sampling_mode"] == "frozen_panel"
    assert first["num_battles"] == 4
    assert first["metrics"]["bradley_terry"]["method"] == "Frozen-anchor Soft-ELO"
    assert entry_a["overall"]["n_battles"] == 4
    assert (
        next(row for row in board_after_b["entries"] if row["model"] == "Dummy/a/b")
        == entry_a
    )
    assert {row["model"] for row in board_after_b["entries"]} == {
        "anchor",
        "Dummy/a/b",
        "Dummy/a_b",
    }
    battles = pd.read_parquet(next((tmp_path / "results").rglob("battles.parquet")))
    assert set(battles["panel_id"]) == {"en-0", "en-1", "fr-0", "fr-1"}


def test_frozen_runner_rejects_changed_config(tmp_path):
    directory = tmp_path / "leaderboard"
    task = get_packaged_task("elo-comparia")
    cfg = _frozen_run_config(tmp_path, directory, "Dummy/candidate")
    _write_frozen_run_files(cfg, directory, task)
    cfg.judge.temperature = 0.5

    with pytest.raises(ValueError, match="runtime config does not match"):
        run_elo(cfg, task)


@pytest.mark.parametrize("swap_mode", ["fixed", "both"])
def test_native_cache_full_and_partial_hits_preserve_frozen_results(
    monkeypatch, tmp_path, swap_mode
):
    from judgearena.cache.sqlite import CompletionCache, JudgementCache
    from judgearena.inference import InferenceResult

    task = get_packaged_task("elo-comparia")
    calls = []
    materialized = []

    class Backend:
        def __init__(self, model):
            self.model = model
            materialized.append(model)

        def batch(self, inputs, **_kwargs):
            prompts = [item.to_messages()[-1].content for item in inputs]
            calls.append((self.model, prompts))
            if self.model == "Dummy/candidate":
                return [f"Candidate response to {prompt}" for prompt in prompts]
            # Different scores expose accidental reordering of cached rows.
            return [
                InferenceResult(
                    text=(
                        "score A: 2 score B: 8"
                        if "Instruction en 0" in prompt
                        else "score A: 7 score B: 4"
                    ),
                    first_token_top_logprobs={"score": -0.1},
                )
                for prompt in prompts
            ]

    monkeypatch.setattr(models, "make_model", lambda model, **_kwargs: Backend(model))
    results = []
    for index in range(3):
        directory = tmp_path / f"leaderboard-{index}"
        cfg = _frozen_run_config(
            tmp_path / f"run-{index}", directory, "Dummy/candidate"
        )
        cfg.judge.swap_mode = swap_mode
        cfg.run.store_root = str(tmp_path / "cache")
        _write_frozen_run_files(cfg, directory, task)
        if index == 2:
            completion_path = next(Path(cfg.run.store_root).rglob("completions.db"))
            judgement_path = next(Path(cfg.run.store_root).rglob("judgements.db"))
            completion_store = CompletionCache(completion_path)
            judgement_store = JudgementCache(judgement_path)
            assert set(completion_store.query().instruction_id) == {
                "en-0",
                "en-1",
                "fr-0",
                "fr-1",
            }
            assert set(judgement_store.query().instruction_id) == {
                "en-0",
                "en-1",
                "fr-0",
                "fr-1",
            }
            assert judgement_store.query().top_logprobs.map(json.loads).tolist() == [
                {"score": -0.1}
            ] * (8 if swap_mode == "both" else 4)
            assert completion_store.delete(instruction_id="en-1") == 1
            assert judgement_store.delete(instruction_id="en-1") == (
                2 if swap_mode == "both" else 1
            )
            completion_store.close()
            judgement_store.close()
        calls.clear()
        materialized.clear()
        results.append(run_elo(cfg, task))
        if index == 1:
            assert calls == []
            assert materialized == []
        elif index == 2:
            assert all(len(prompts) == 1 for _, prompts in calls)
            assert len(calls) == (3 if swap_mode == "both" else 2)
            assert all("Instruction en 1" in prompts[0] for _, prompts in calls)

    first_dir = Path(results[0]["result_path"]).parent
    for result in results[1:]:
        directory = Path(result["result_path"]).parent
        assert json.loads((directory / "entry.json").read_text()) == json.loads(
            (first_dir / "entry.json").read_text()
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(directory / "battles.parquet"),
            pd.read_parquet(first_dir / "battles.parquet"),
        )
    battles = pd.read_parquet(first_dir / "battles.parquet")
    assert tuple(battles.columns) == BATTLE_COLUMNS
    assert battles.panel_id.tolist() == ["en-0", "en-1", "fr-0", "fr-1"] * (
        2 if swap_mode == "both" else 1
    )


@pytest.mark.parametrize("swap_mode", ["fixed", "both"])
def test_frozen_runner_keeps_malformed_raw_rows_but_skips_whole_comparison(
    monkeypatch, tmp_path, swap_mode
):
    task = get_packaged_task("elo-comparia")
    directory = tmp_path / "leaderboard"
    cfg = _frozen_run_config(tmp_path, directory, "Dummy/candidate")
    cfg.judge.swap_mode = swap_mode
    _write_frozen_run_files(cfg, directory, task)
    judge_pass = 0

    def batch(self, inputs, **_kwargs):
        nonlocal judge_pass
        if self.name == cfg.model.name:
            return ["candidate response"] * len(inputs)
        judge_pass += 1
        outputs = ["score A: 2 score B: 8"] * len(inputs)
        if judge_pass == 1:
            outputs[0] = "malformed"
        return outputs

    monkeypatch.setattr(models.DummyModel, "batch", batch)
    result = run_elo(cfg, task)
    output_dir = Path(result["result_path"]).parent
    battles = pd.read_parquet(output_dir / "battles.parquet")
    assert len(battles) == (8 if swap_mode == "both" else 4)
    assert battles.pref.isna().sum() == 1
    assert result["sampling_metadata"]["skipped_battles"] == 1
    entry = LeaderboardEntry.model_validate_json(
        (output_dir / "entry.json").read_text()
    )
    assert entry.overall.n_battles == 3
    assert entry.by_language["en"].n_battles == 1


def test_frozen_runner_hardens_each_pass_before_collapsing(monkeypatch, tmp_path):
    task = get_packaged_task("elo-comparia")
    directory = tmp_path / "leaderboard"
    cfg = _frozen_run_config(tmp_path, directory, "Dummy/candidate")
    cfg.judge.swap_mode = "both"
    cfg.elo.soft_elo = False
    _write_frozen_run_files(cfg, directory, task)
    judge_pass = 0

    def batch(self, inputs, **_kwargs):
        nonlocal judge_pass
        if self.name == cfg.model.name:
            return ["candidate response"] * len(inputs)
        judge_pass += 1
        return [
            "score A: 0 score B: 10" if judge_pass == 1 else "score A: 4 score B: 5"
        ] * len(inputs)

    monkeypatch.setattr(models.DummyModel, "batch", batch)
    result = run_elo(cfg, task)
    battles = pd.read_parquet(Path(result["result_path"]).parent / "battles.parquet")
    collapsed = collapse_swapped_rows(battles, "both")
    assert collapsed.pref_hard.tolist() == [0.5] * 4
    assert collapsed.pref.gt(0.5).all()
    assert result["metrics"]["bradley_terry"]["rating"] == pytest.approx(1000)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("elo", "languages", ["en"], "exactly match"),
        ("generation", "n_instructions", 1, "cannot resample"),
        ("elo", "n_instructions_per_language", 1, "cannot resample"),
        ("elo", "elo_random_battles", 1, "cannot resample"),
        ("elo", "calibrate_temperature", True, "cannot recalibrate"),
        ("model", "name", "anchor", "already on the leaderboard"),
    ],
)
def test_frozen_preflight_rejects_invalid_runs_before_inference(
    monkeypatch, tmp_path, section, field, value, message
):
    task = get_packaged_task("elo-comparia")
    directory = tmp_path / "leaderboard"
    cfg = _frozen_run_config(tmp_path, directory, "Dummy/candidate")
    _write_frozen_run_files(cfg, directory, task)
    setattr(getattr(cfg, section), field, value)
    monkeypatch.setattr(
        models,
        "make_model",
        lambda *_args, **_kwargs: pytest.fail("invalid frozen run reached inference"),
    )
    with pytest.raises(ValueError, match=message):
        run_elo(cfg, task)
