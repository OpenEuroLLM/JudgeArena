import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import judgearena.benchmarks.evaluation as pairwise_runner
import judgearena.benchmarks.pairwise.calibration as elo_calibration
import judgearena.benchmarks.pairwise.comparisons as preparation
import judgearena.models as models
from judgearena.benchmarks.elo.rating import (
    arena_anchor_battles,
    fit_bradley_terry,
)
from judgearena.benchmarks.elo.scoring import BradleyTerryMetric
from judgearena.benchmarks.runner import run_benchmark
from judgearena.cache.sqlite import JudgementCache
from judgearena.config import JudgePromptSpec, RunConfig, build_run_config, load_config
from judgearena.datasets._arena import normalize_battles
from judgearena.datasets.data import PreferenceData
from judgearena.models import DummyModel, make_model
from judgearena.prompts.registry import resolve_judge_prompt
from judgearena.tasks.registry import get_packaged_task

N_BATTLES = 30
ARENA_MODELS = ["arena_model_alpha", "arena_model_beta", "arena_model_gamma"]


def _make_conversation(content_user: str, content_assistant: str) -> list[dict]:
    return [
        {"role": "user", "content": content_user},
        {"role": "assistant", "content": content_assistant},
    ]


def _arena_df(n_battles: int) -> pd.DataFrame:
    """Synthetic arena DataFrame matching the schema read by the arena adapter."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_battles):
        ma, mb = rng.choice(ARENA_MODELS, size=2, replace=False)
        winner = rng.choice(["model_a", "model_b", "tie"])
        lang = rng.choice(["en", "fr"])
        rows.append(
            {
                "question_id": f"q{i}",
                "tstamp": 1700000000 + i,
                "model_a": ma,
                "model_b": mb,
                "winner": winner,
                "conversation_a": _make_conversation(
                    f"Instruction {i}", f"Response A {i}"
                ),
                "conversation_b": _make_conversation(
                    f"Instruction {i}", f"Response B {i}"
                ),
                "benchmark": "TestArena",
                "lang": lang,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_arena_df() -> pd.DataFrame:
    return _arena_df(N_BATTLES)


@pytest.fixture(autouse=True)
def mock_external_deps(monkeypatch, synthetic_arena_df):
    monkeypatch.setattr(
        pairwise_runner,
        "load_task_data",
        lambda _task: normalize_battles(synthetic_arena_df, "ComparIA"),
    )

    def mock_generate(instructions, model, **kwargs):
        return pd.DataFrame(
            {
                "completion": [
                    f"Synthetic completion {i}" for i in range(len(instructions))
                ],
                "instruction_index": instructions.index,
            }
        )

    monkeypatch.setattr(preparation, "generate_instructions", mock_generate)


def _default_args(*, result_folder: str, **kwargs) -> RunConfig:
    task = kwargs.pop("task", "elo-comparia")
    model = kwargs.pop("model", "Dummy/my model")
    judge_model = kwargs.pop("judge_model", "Dummy/score A: 0 score B: 10")
    n_instructions = kwargs.pop("n_instructions", 10)
    n_bootstraps = kwargs.pop("n_bootstraps", 3)
    languages = kwargs.pop("languages", None)
    swap_mode = kwargs.pop("swap_mode", "fixed")
    strip_thinking_before_judging = kwargs.pop("strip_thinking_before_judging", False)
    calibrate_temperature = kwargs.pop("calibrate_temperature", False)
    store_root = kwargs.pop("store_root", None)
    assert not kwargs, f"unexpected kwargs: {kwargs}"
    judge: dict[str, object] = {
        "model": judge_model,
        "swap_mode": swap_mode,
        "strip_thinking_before_judging": strip_thinking_before_judging,
    }
    return RunConfig(
        task=task,
        model={"name": model},
        judge=judge,
        protocol={
            "judge": {"calibrate_temperature": calibrate_temperature},
            "scoring": {
                "metrics": [
                    {"metric": "pairwise_win_rate"},
                    {
                        "metric": "bradley_terry",
                        "parameters": {
                            "fit_mode": "frozen",
                            "soft": True,
                            "n_bootstraps": n_bootstraps,
                        },
                    },
                ]
            },
        },
        run={
            "sampling": {"limit": n_instructions},
            "result_folder": result_folder,
            "store_root": store_root,
            "filters": {"lang": languages} if languages else {},
        },
    )


def test_bradley_terry_hard_mode_uses_hard_preferences():
    battles = pd.DataFrame(
        {
            "model_a": ["candidate", "candidate", "opponent", "opponent"],
            "model_b": ["opponent", "opponent", "candidate", "candidate"],
            "pref": [0.5, 0.5, 0.5, 0.5],
            "pref_hard": [0.0, 0.0, 1.0, 1.0],
            "source": ["llm-judge"] * 4,
            "evaluation_model": ["candidate"] * 4,
        }
    )

    soft = BradleyTerryMetric(soft=True).calculate(battles)
    hard = BradleyTerryMetric(soft=False).calculate(battles)

    assert soft["ratings"]["candidate"] == pytest.approx(soft["ratings"]["opponent"])
    assert hard["ratings"]["candidate"] > hard["ratings"]["opponent"]


def test_bradley_terry_metric_reports_anchor_and_bootstrap_results():
    battles = pd.DataFrame(
        {
            "model_a": ["anchor-a", "anchor-b", "candidate", "anchor-a"],
            "model_b": ["anchor-b", "anchor-a", "anchor-a", "candidate"],
            "pref": [0.0, 1.0, 0.2, 0.8],
            "pref_hard": [0.0, 1.0, 0.0, 1.0],
            "reference_pref": [0.0, 1.0, None, None],
            "evaluation_model": [None, None, "candidate", "candidate"],
        },
        index=[0, 1, 0, 1],
    )
    result = BradleyTerryMetric(n_bootstraps=3, baseline_model="anchor-b").calculate(
        battles, seed=17
    )
    assert result["ratings"]["anchor-b"] == pytest.approx(1000)
    assert set(result["reference_ratings"]) == {"anchor-a", "anchor-b"}
    assert len(result["bootstrap_ratings"]) == result["n_bootstraps"] == 3
    assert result["evaluation_model"] == "candidate"
    assert result["battle_counts"]["candidate"] == 2
    assert {entry["model"] for entry in result["rating_entries"]} == {
        "anchor-a",
        "anchor-b",
        "candidate",
    }


@pytest.mark.parametrize(
    "preferences, expected_tie", [([0, 1], False), ([0.5, 0.5], True)]
)
def test_joint_fit_anchors_opponent(preferences, expected_tie):
    battles = pd.DataFrame(
        {"model_a": ["A", "B"], "model_b": ["B", "A"], "pref": preferences}
    )
    ratings = fit_bradley_terry(battles, baseline_model="B")
    assert ratings["B"] == pytest.approx(1000)
    if expected_tie:
        assert ratings["A"] == pytest.approx(1000)
    else:
        assert ratings["A"] > ratings["B"]


def run_graph_evaluation(cfg: RunConfig, task=None) -> dict:
    cfg = cfg.for_task(task)
    manifest = run_benchmark(cfg, output_dir=Path(cfg.run.result_folder) / "evaluation")
    report = manifest.parent / json.loads(manifest.read_text())["artifacts"]["report"]
    return {**json.loads(report.read_text()), "result_path": str(report)}


def _pairwise_metric(run_result: dict) -> dict:
    return run_result["metrics"]["pairwise_win_rate"]


@pytest.mark.parametrize("task", ["alpaca-eval-ja", "elo-comparia"])
def test_random_instructions_compare_each_selected_prompt_to_all_baselines(
    tmp_path, synthetic_arena_df, monkeypatch, task
):
    data = normalize_battles(synthetic_arena_df, "ComparIA")
    metrics = [{"metric": "pairwise_win_rate", "breakdown_by": ["baseline"]}]
    if task == "elo-comparia":
        metrics.append(
            {"metric": "bradley_terry", "parameters": {"fit_mode": "frozen"}}
        )
    cfg = RunConfig(
        task=task,
        model={"name": "Dummy/candidate", "baseline": ARENA_MODELS[:2]},
        judge={"model": "Dummy/score A: 0 score B: 10", "swap_mode": "fixed"},
        protocol={"scoring": {"metrics": metrics}},
        run={
            "sampling": {"method": "random", "limit": 20},
            "seed": 17,
            "store_root": str(tmp_path / "cache"),
        },
    )
    run_benchmark(cfg, data=data, output_dir=tmp_path / "evaluation")
    if task == "elo-comparia":
        report = json.loads((tmp_path / "evaluation/results.json").read_text())
        rating = report["metrics"]["bradley_terry"]
        assert set(rating["reference_ratings"]) == set(ARENA_MODELS), rating
    rows = pd.read_parquet(tmp_path / "evaluation/outcomes.parquet")
    expected = set(data.instructions.sample(20, random_state=17).instruction)
    assert len(rows) == 40
    assert {
        opponent: set(part.instruction) for opponent, part in rows.groupby("model_b")
    } == {opponent: expected for opponent in ARENA_MODELS[:2]}
    monkeypatch.setattr(
        models.PreparedModel, "materialize", lambda *_: pytest.fail("cache miss")
    )
    saved = load_config(tmp_path / "evaluation/config.yaml")
    run_benchmark(saved, data=data, output_dir=tmp_path / "replay")
    pd.testing.assert_frame_equal(
        rows, pd.read_parquet(tmp_path / "replay/outcomes.parquet")
    )


def test_run_elo_without_bradley_terry_skips_rating_artifacts(tmp_path):
    cfg = _default_args(result_folder=str(tmp_path))
    values = cfg.model_dump()
    values["protocol"] = {
        "scoring": {
            "metrics": [{"metric": "pairwise_win_rate", "breakdown_by": ["missing"]}]
        }
    }
    cfg = RunConfig(**values)
    result = run_graph_evaluation(cfg)

    assert set(result["metrics"]) == {"pairwise_win_rate"}
    # Graph runs use the same saved-answer and evaluation artifacts as other tasks.
    assert len(pd.read_parquet(next(tmp_path.rglob("answers.parquet")))) == 10
    manifest = next(tmp_path.rglob("evaluation.json"))
    from judgearena.artifacts.evaluation import metric_data_checksum

    checksum_data = pd.DataFrame({"pref": [0.5, 0.75], "model": ["a", "b"]})
    checksum_path = tmp_path / "checksum.parquet"
    checksum_data.to_parquet(checksum_path, index=False)
    original = metric_data_checksum(checksum_path)
    checksum_data[["model", "pref"]].to_parquet(
        checksum_path, index=False, compression=None
    )
    assert metric_data_checksum(checksum_path) == original
    checksum_data.loc[0, "pref"] = np.nextafter(0.5, 1.0)
    checksum_data.to_parquet(checksum_path, index=False)
    assert metric_data_checksum(checksum_path) != original

    metadata = json.loads((manifest.parent / "run-metadata.v1.json").read_text())
    assert all(
        (manifest.parent / item["path"]).is_file() for item in metadata["artifacts"]
    )
    assert not list(tmp_path.rglob("bootstrap_ratings.csv"))
    assert not list(tmp_path.rglob("elo_ratings.json"))
    log_text = next(tmp_path.rglob("run-*.log")).read_text()
    assert "Skipping metric 'pairwise_win_rate' breakdown 'missing'" in log_text


@pytest.mark.parametrize(
    ("swap_mode", "strip_thinking"),
    [("fixed", False), ("random", True), ("both", True)],
)
def test_run_elo_aligns_shuffled_completions_with_judged_battles(
    monkeypatch, tmp_path, synthetic_arena_df, swap_mode, strip_thinking
):
    for slot in ("a", "b"):
        synthetic_arena_df[f"messages_{slot}"] = synthetic_arena_df[
            f"conversation_{slot}"
        ]
    synthetic_arena_df["checkpoint_a"] = "source-a"
    synthetic_arena_df["checkpoint_b"] = "source-b"
    synthetic_arena_df["topic"] = ["coding", "math", "writing"] * (N_BATTLES // 3)
    battles = normalize_battles(synthetic_arena_df, "ComparIA").battles
    data = PreferenceData.from_battles(
        battles.drop(
            columns=["conversation_a", "conversation_b", "question_id", "source"]
        )
    )
    data.battles.loc[1, "reference_pref"] = float("nan")
    preloaded = pd.DataFrame(
        [
            {
                "instruction_index": 0,
                "model": "Dummy/my model",
                "completion": "<think>private 0</think>Answer 0",
                "checkpoint": "saved-candidate",
            }
        ]
    )
    monkeypatch.setattr(
        pairwise_runner,
        "load_task_data",
        lambda _: replace(
            data, model_outputs=pd.concat([data.model_outputs, preloaded])
        ),
    )

    def generate(instructions, **_kwargs):
        assert instructions.tolist() == ["Instruction 1", "Instruction 2"]
        return pd.DataFrame(
            {
                "instruction_index": instructions.index,
                "completion": [f"<think>private {i}</think>Answer {i}" for i in (1, 2)],
            }
        ).iloc[::-1]

    monkeypatch.setattr(preparation, "generate_instructions", generate)
    cfg = _default_args(
        result_folder=str(tmp_path),
        store_root=str(tmp_path / "cache"),
        n_instructions=3,
        n_bootstraps=0,
        swap_mode=swap_mode,
        strip_thinking_before_judging=strip_thinking,
    )
    cfg.run.seed = 42
    run_graph_evaluation(cfg)

    with JudgementCache(next((tmp_path / "cache").rglob("judgements.db"))) as cache:
        cached = cache.query()
    assert len(cached) == (6 if swap_mode == "both" else 3)
    for row in cached.itertuples():
        i = int(row.instruction_id.rsplit("q", 1)[1])
        assert f"Answer {i}" in row.judge_input
        opponent_slot = ("a", "b", "b")[i]
        expected_model = synthetic_arena_df.iloc[i][f"model_{opponent_slot}"]
        assert expected_model == (
            row.model_b if row.model_a == cfg.model.name else row.model_a
        )
        assert f"Response {opponent_slot.upper()} {i}" in row.judge_input
        assert ("private" in row.judge_input) is not strip_thinking
        assert row.orientation == (
            "direct" if row.model_a == cfg.model.name else "reversed"
        )

    battles = pd.read_parquet(next(tmp_path.rglob("outcomes.parquet")))
    repeats = 2 if swap_mode == "both" else 1
    assert battles["question_id"].tolist() == [0, 1, 2] * repeats
    assert battles["winner"].notna().all()
    assert (battles["model_a"] == cfg.model.name).tolist() == [
        True,
        True,
        False,
    ] * repeats
    direct_pref = 1 / (
        1 + math.exp(-cfg.preference_settings.preference_temperature * 10)
    )
    expected = [direct_pref] * 3
    if swap_mode == "both":
        expected += [1 - direct_pref] * 3
    assert battles["pref"].tolist() == pytest.approx(expected)
    annotations = pd.read_parquet(next(tmp_path.rglob("annotations.parquet")))
    assert "reference_pref" not in annotations
    assert annotations["topic"].tolist() == ["coding", "math", "writing"] * repeats
    assert (
        annotations["lang"].tolist()
        == synthetic_arena_df["lang"].iloc[:3].tolist() * repeats
    )
    assert annotations["source"].eq("llm-judge").all()
    assert "winner" not in annotations
    for row in annotations.itertuples():
        candidate = "a" if row.model_a == cfg.model.name else "b"
        opponent = "b" if candidate == "a" else "a"
        checkpoint = getattr(row, f"checkpoint_{candidate}")
        assert (
            checkpoint == "saved-candidate"
            if row.question_id == 0
            else pd.isna(checkpoint)
        )
        assert getattr(row, f"checkpoint_{opponent}").startswith("source-")
        assert getattr(row, f"messages_{opponent}") is not None


def test_run_elo_keeps_battles_with_repeated_question_ids(tmp_path, synthetic_arena_df):
    synthetic_arena_df.loc[1, "question_id"] = "q0"
    run_graph_evaluation(
        _default_args(
            result_folder=str(tmp_path),
            n_instructions=3,
            n_bootstraps=0,
            swap_mode="both",
        )
    )
    battles = pd.read_parquet(next(tmp_path.rglob("outcomes.parquet")))
    assert battles["question_id"].tolist() == ["q0", "q0", "q2"] * 2


def test_arena_anchor_battles_filters_and_preserves_index():
    # Anchors are rebuilt on recompute, so this primitive must drop under-
    # represented models (< 500 battles), retain reference labels, and preserve the
    # arena row labels (calibration looks up conversations via df_arena_all.loc[i]).
    n = 500
    df_all = pd.DataFrame(
        {
            "model_a": ["x"] * n + ["rare"],
            "model_b": ["y"] * n + ["x"],
            "reference_pref": [0.0, 1.0] * (n // 2) + [0.0],
            "source": "human",
            "conversation_a": [["q"]] * (n + 1),  # extra column must be ignored
        },
        index=range(1000, 1000 + n + 1),
    )
    out = arena_anchor_battles(df_all)

    # x, y have >= 500 battles -> kept; 'rare' (1 battle) -> its row dropped
    assert set(out["model_a"]) | set(out["model_b"]) == {"x", "y"}
    assert list(out.index) == list(range(1000, 1000 + n))  # labels preserved, rare gone
    assert out["reference_pref"].equals(out["pref"])
    assert out.loc[1000, "pref"] == 0.0 and out.loc[1001, "pref"] == 1.0


@pytest.mark.parametrize(
    "task, languages", [("elo-lmarena-140k-en", None), ("elo-lmarena-140k", ["en"])]
)
def test_elo_language_variant_resolves_and_filters(
    tmp_path, synthetic_arena_df, task, languages
):
    cfg = _default_args(
        result_folder=str(tmp_path), task=task, languages=languages, n_instructions=2
    )
    run_graph_evaluation(cfg)
    output = tmp_path / "evaluation"
    assert pd.read_parquet(output / "outcomes.parquet").lang.tolist() == ["en", "en"]
    references = pd.read_parquet(output / "reference-battles.parquet")
    assert len(references) == synthetic_arena_df.lang.eq("en").sum() > 2
    assert set(references.lang) == {"en"}


@pytest.mark.parametrize("swap_mode", ["fixed", "both"])
def test_run_elo_calibration_reuses_judge_and_cache(monkeypatch, tmp_path, swap_mode):
    from unittest.mock import Mock

    finish = Mock(wraps=pairwise_runner.finish_evaluation)
    monkeypatch.setattr(pairwise_runner, "finish_evaluation", finish)
    captured = {}
    loaded_models = []

    def load_model(model, **kwargs):
        loaded_models.append(model)
        return make_model(model, **kwargs)

    monkeypatch.setattr(models, "make_model", load_model)

    def fake_calibrate(delta_s, y):
        captured["n_pairs"] = len(delta_s)
        return 0.42

    monkeypatch.setattr(elo_calibration, "fit_temperature", fake_calibrate)
    # Anchor battles require models with >= 500 appearances; the default
    # 30-battle fixture leaves the calibration pool empty.
    arena = _arena_df(901)
    arena.loc[900, ["model_a", "model_b", "winner"]] = ["rare-a", "rare-b", "model_a"]
    monkeypatch.setattr(
        pairwise_runner,
        "load_task_data",
        lambda _task: normalize_battles(arena, "ComparIA"),
    )

    store_root = tmp_path / "cache"
    cfg = _default_args(
        result_folder=str(tmp_path),
        calibrate_temperature=True,
        swap_mode=swap_mode,
        store_root=str(store_root),
    )
    result = run_graph_evaluation(cfg)

    assert captured["n_pairs"] == (arena.iloc[:900]["winner"] != "tie").sum()
    assert loaded_models == ["Dummy/score A: 0 score B: 10"]
    assert 0.0 <= _pairwise_metric(result)["winrate"] <= 1.0
    battles = pd.read_parquet(next(tmp_path.rglob("outcomes.parquet")))
    pref = 1 / (1 + math.exp(-0.42 * 10))
    expected = [pref] * 10 + ([1 - pref] * 10 if swap_mode == "both" else [])
    assert battles["pref"].tolist() == pytest.approx(expected)
    judgement_db = next((store_root / "judgements").rglob("judgements.db"))
    with JudgementCache(judgement_db) as cache:
        assert cache.query()["instruction_id"].str.startswith("ComparIA:").all()

    def unavailable(*args, **kwargs):
        raise AssertionError("Cached judging and calibration must not load a model")

    monkeypatch.setattr(models, "make_model", unavailable)
    cfg = build_run_config(
        [
            "--config",
            str(Path(result["result_path"]).parent / "config.yaml"),
            "--protocol.judge.calibrate_temperature",
            "yes",
        ]
    )
    cfg.run.result_folder = str(tmp_path / "replay")
    replay = run_graph_evaluation(cfg)
    assert _pairwise_metric(replay) == _pairwise_metric(result)
    assert finish.call_count == 2  # One finalization per live/replay run.
    replay_dir = Path(replay["result_path"]).parent
    pd.testing.assert_frame_equal(
        battles, pd.read_parquet(replay_dir / "outcomes.parquet")
    )
    manifest = json.loads((replay_dir / "evaluation.json").read_text())
    assert "outcomes.parquet" in manifest["files"]
    assert not (replay_dir / "battles.parquet").exists()


def test_empty_calibration_keeps_configured_temperature(tmp_path):
    run_graph_evaluation(
        _default_args(result_folder=str(tmp_path), calibrate_temperature=True)
    )
    battles = pd.read_parquet(next(tmp_path.rglob("outcomes.parquet")))
    assert battles["pref"].tolist() == pytest.approx([0.9525741268224334] * 10)


def test_extract_turn_text_tolerates_moderated_turns():
    from judgearena.datasets._arena import extract_turn_text

    assert extract_turn_text({"content": None}) == ""
    assert extract_turn_text({"content": "plain"}) == "plain"
    assert (
        extract_turn_text(
            {"content": [{"type": "text", "text": None}, {"type": "image"}, None]}
        )
        == ""
    )


def test_run_elo_preserves_soft_preferences_from_non_pairscore_parser(tmp_path):
    cfg = _default_args(
        result_folder=str(tmp_path),
        store_root=str(tmp_path / "cache"),
        judge_model="Dummy/[[B>A]]",
    )
    cfg.judge.prompt = JudgePromptSpec(preset="arena-hard")

    run_graph_evaluation(cfg)

    battles = pd.read_parquet(next(tmp_path.rglob("outcomes.parquet")))
    assert battles["pref"].tolist() == [0.75] * 10


def test_run_elo_temperature_reuses_validated_scores(monkeypatch, tmp_path):
    from unittest.mock import Mock

    from judgearena.prompts.parsing import JUDGE_PARSERS

    parser = JUDGE_PARSERS["meta-eval-score"]
    parse = Mock(wraps=parser.parse_result)
    monkeypatch.setattr(parser, "parse_result", parse)
    monkeypatch.setattr(
        DummyModel,
        "batch",
        lambda self, inputs, **kwargs: [
            "Score_A: 6\nScore_B: 8",
            "Score_A: 6\nScore_B: missing",
        ],
    )
    cfg = _default_args(result_folder=str(tmp_path), n_instructions=2, swap_mode="both")
    cfg.judge.prompt = JudgePromptSpec(preset="meta-eval-pair-score")
    cfg = RunConfig(
        **{
            **cfg.model_dump(),
            "protocol": {"judge": {"preference_temperature": 0.8}},
        }
    )

    result = run_graph_evaluation(cfg)

    battles = pd.read_parquet(next(tmp_path.rglob("outcomes.parquet")))
    assert battles["pref"].tolist() == pytest.approx(
        [0.8320183851339245, float("nan"), 0.1679816148660755, float("nan")],
        nan_ok=True,
    )
    assert battles["pref"].isna().sum() == 2
    assert _pairwise_metric(result)["num_missing"] == 2
    assert parser.temperature == 0.5
    assert parse.call_count == len(battles)
    annotations = pd.read_parquet(next(tmp_path.rglob("annotations.parquet")))
    assert annotations["pref"].tolist() == pytest.approx(
        [0.7310585786, float("nan"), 0.2689414214, float("nan")], nan_ok=True
    )
    assert annotations["judge"].eq(cfg.judge.model).all()


@pytest.mark.parametrize(
    "invalid", ["missing", "delegated", "scalar", "mode", "logprobs"]
)
def test_run_elo_rejects_unusable_prompt_before_generation(
    monkeypatch, tmp_path, invalid
):
    prompt = resolve_judge_prompt(preset="default")
    if invalid == "missing":
        prompt = replace(prompt, parser=None)
    elif invalid == "delegated":
        prompt = replace(prompt, delegated=True)
    elif invalid == "scalar":
        prompt = replace(
            prompt, parser=resolve_judge_prompt(preset="wildbench-score").parser
        )
    elif invalid == "logprobs":
        prompt = resolve_judge_prompt(preset="alpaca-eval")
    else:
        prompt = replace(prompt, judging_mode="single_answer")
    monkeypatch.setattr(
        pairwise_runner,
        "resolve_run_judge_prompt",
        lambda *_args, **kwargs: prompt,
    )
    monkeypatch.setattr(
        preparation,
        "generate_instructions",
        lambda *_args, **_kwargs: pytest.fail(
            "Prompt validation must precede generation"
        ),
    )
    with pytest.raises(
        ValueError,
        match="parser|ParsedPreference|ParsedScore|judging mode|top_logprobs|delegated",
    ):
        run_graph_evaluation(_default_args(result_folder=str(tmp_path)))


def test_elo_prepares_history_answers_once_and_replays(monkeypatch, tmp_path):
    task = get_packaged_task("elo-comparia")
    spec = task.spec.model_dump()
    spec["protocol"]["generation"]["mode"] = "provided_messages"
    task = replace(task, spec=type(task.spec).model_validate(spec))
    rows = pd.DataFrame(
        {
            "instruction_index": [1, 1, 2, 3],
            "instruction": ["Q1", "Q1", "Q2", "Q3"],
            "instruction_id": ["cache:1", "cache:1", "cache:2", "cache:3"],
            "model_a": "a",
            "model_b": ["b", "c", "b", "c"],
            "completion_a": "A",
            "completion_b": "B",
            "reference_pref": 0.5,
        }
    )
    rows["messages"] = [
        [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": q},
        ]
        for q in rows["instruction"]
    ]
    data = PreferenceData.from_battles(rows)
    data = replace(
        data,
        model_outputs=pd.DataFrame(
            {
                "instruction_index": [1, 2, 3],
                "model": "Dummy/generated",
                "completion": [None, "", "saved"],
                "rubric": ["Grammar"] * 3,
            }
        ),
    )
    monkeypatch.setattr(pairwise_runner, "load_task_data", lambda _: data)
    prompt = resolve_judge_prompt(preset="default")
    prompt = replace(
        prompt,
        user_prompt_template=prompt.user_prompt_template
        + "\nID: {question_id} Rubrics: {rubric_a} {rubric_b}"
        + "\nConversations: {messages_a} {messages_b}",
    )
    monkeypatch.setattr(
        pairwise_runner, "resolve_run_judge_prompt", lambda *_args, **kwargs: prompt
    )
    seen = []
    original_batch = DummyModel.batch

    def capture(self, inputs, **kwargs):
        if self.name == "Dummy/generated":
            seen.extend(inputs)
        return original_batch(self, inputs, **kwargs)

    monkeypatch.setattr(DummyModel, "batch", capture)
    cfg = _default_args(
        result_folder=str(tmp_path / "first"),
        store_root=str(tmp_path / "cache"),
        model="Dummy/generated",
        n_instructions=None,
        n_bootstraps=0,
    )
    cfg = RunConfig(
        **{
            **cfg.model_dump(),
            "protocol": {"generation": task.spec.protocol.generation.model_dump()},
        }
    )
    first = run_graph_evaluation(cfg)
    assert len(seen) == 1
    assert [m.content for m in seen[0].to_messages()] == [
        "Earlier question",
        "Earlier answer",
        "Q1",
    ]
    annotations = pd.read_parquet(
        next((tmp_path / "first").rglob("annotations.parquet"))
    )
    answers = [
        getattr(
            row, "completion_a" if row.model_a == cfg.model.name else "completion_b"
        )
        for row in annotations.itertuples()
    ]
    assert answers == ["generated", "generated", "", "saved"]
    monkeypatch.setattr(
        models, "make_model", lambda *a, **kw: pytest.fail("Replay must use cache")
    )
    cfg.run.result_folder = str(tmp_path / "replay")
    replay = run_graph_evaluation(cfg, task)
    assert _pairwise_metric(replay) == _pairwise_metric(first)


@pytest.mark.parametrize(
    "default, category", [("default", "arena-hard"), ("arena-hard", "default")]
)
def test_elo_temperature_follows_category_parser(
    monkeypatch, tmp_path, default, category
):
    task = get_packaged_task("elo-comparia")
    spec = task.spec.model_dump()
    spec["protocol"]["judge"].update(
        default_prompt_preset=default, category_prompts={"special": category}
    )
    task = replace(task, spec=type(task.spec).model_validate(spec))
    data = normalize_battles(
        _arena_df(2).assign(category=["regular", "special"]), "ComparIA"
    )
    monkeypatch.setattr(pairwise_runner, "load_task_data", lambda task: data)
    cfg = _default_args(
        result_folder=str(tmp_path),
        n_instructions=2,
        judge_model="Dummy/Score_A: 6 Score_B: 8 [[A>>B]]",
    )
    cfg = RunConfig(
        **{
            **cfg.model_dump(),
            "protocol": {"judge": {"preference_temperature": 0.8}},
        }
    )
    run_graph_evaluation(cfg, task)
    rows = pd.read_parquet(next(tmp_path.rglob("outcomes.parquet")))
    annotations = pd.read_parquet(next(tmp_path.rglob("annotations.parquet")))
    for row, pref in zip(annotations.itertuples(), rows.pref, strict=True):
        preference = 1 / (1 + math.exp(-1.6)) if row.prompt_preset == "default" else 0.0
        assert pref == pytest.approx(1 - preference if row.swapped else preference)


def test_sampling_quotas_and_limit_keep_complete_dialogues():
    from judgearena.config import SamplingArgs

    rows = pd.DataFrame(
        {"dialogue": [1, 1, 2, 3, 3, 4], "lang": ["de"] * 3 + ["en"] * 3}
    )
    quota = SamplingArgs(group_by="lang", per_group=1)
    assert preparation.select_rows(
        rows, quota, seed=0, limit_by="dialogue"
    ).dialogue.tolist() == [1, 1, 3, 3]
    quota.limit = 1
    assert preparation.select_rows(
        rows, quota, seed=0, limit_by="dialogue"
    ).dialogue.tolist() == [1, 1]


@pytest.mark.parametrize("temperature", [0.7, None])
def test_graph_sampling_and_judge_settings_do_not_depend_on_metrics(
    tmp_path, monkeypatch, temperature
):
    from judgearena.artifacts import read_metric_data
    from judgearena.benchmarks.runner import run_benchmark
    from judgearena.benchmarks.scoring import _METRIC_TYPES

    class CountRows:
        def calculate(self, battles):
            return {"count": len(battles)}

        def render(self, values):
            return str(values["count"])

    monkeypatch.setitem(_METRIC_TYPES, "count_rows", CountRows)
    cfg = RunConfig(
        task="elo-comparia",
        model={"name": "Dummy/candidate"},
        judge={
            "model": "Dummy/score A: 6 score B: 8",
            "prompt": {
                "user": "{rubric}: {completion_a} or {completion_b}?",
                "parser": "score",
            },
        },
        protocol={
            "runner": "pairwise",
            "limit_by": "dialogue",
            "generation": {"mode": "single_turn_chat"},
            "baseline": {"strategy": "recorded_opponents"},
            "judge": {
                "default_prompt_preset": "default",
                **(
                    {"preference_temperature": temperature}
                    if temperature is not None
                    else {}
                ),
                "calibrate_temperature": False,
            },
            "scoring": {
                "metrics": [{"metric": "pairwise_win_rate"}, {"metric": "count_rows"}]
            },
        },
        run={
            "store_root": str(tmp_path / "cache"),
            "no_log_file": True,
            "sampling": {"group_by": "rubric", "per_group": 3},
        },
    )
    data = normalize_battles(_arena_df(30), "ComparIA")
    data = replace(
        data,
        instructions=data.instructions.assign(
            rubric="clarity", dialogue=np.arange(30) // 2
        ),
    )
    first = run_benchmark(cfg, data=data, output_dir=tmp_path / "first")
    manifest = json.loads(first.read_text())
    artifacts = manifest["artifacts"]
    report = json.loads((first.parent / artifacts["report"]).read_text())
    assert report["report_type"] == "ComparisonReport"
    assert report["arena"] == "elo-comparia"
    assert report["metrics"]["count_rows"] == {"count": 6}
    judgments = read_metric_data(first.parent / artifacts["metric_data"])
    references = read_metric_data(first.parent / artifacts["reference_battles"])
    assert len(judgments) == 6 and len(references) == 30
    assert {"pref", "pref_hard", "completion_a"}.isdisjoint(references)
    assert references.reference_pref.notna().all()
    assert references.rubric.eq("clarity").all()
    assert "winner" in references
    assert (
        judgments.winner.tolist()
        == judgments.pref_hard.map({0: "model_a", 0.5: "tie", 1: "model_b"}).tolist()
    )
    assert "winner" not in pd.read_parquet(first.parent / "annotations.parquet")
    metric = {
        "metric": "bradley_terry",
        "parameters": {"soft": False, "fit_mode": "frozen"},
        "breakdown_by": ["lang", "judge"],
    }
    saved = build_run_config(
        [
            "--config",
            str(first.parent / "config.yaml"),
            "--protocol.scoring.metrics",
            json.dumps([metric]),
        ]
    )
    monkeypatch.setattr(
        models, "make_model", lambda *a, **kw: pytest.fail("Replay must use cache")
    )
    second = run_benchmark(saved, data=data, output_dir=tmp_path / "second")
    pd.testing.assert_frame_equal(
        pd.read_parquet(first.parent / "outcomes.parquet"),
        pd.read_parquet(second.parent / "outcomes.parquet"),
    )
    ratings = json.loads(second.read_text())["metrics"]["bradley_terry"]
    counts = {
        entry["group"]: entry["values"]["num_comparisons"]
        for entry in ratings["groups"]["lang"]
    }
    assert counts == judgments.lang.value_counts().to_dict()
    (judge_group,) = ratings["groups"]["judge"]
    assert judge_group["values"]["ratings"] == ratings["ratings"]
    unlabeled = replace(data, battles=data.battles.drop(columns="reference_pref"))
    third = run_benchmark(cfg, data=unlabeled, output_dir=tmp_path / "unlabeled")
    assert (
        json.loads(third.read_text())["metrics"]
        == json.loads(first.read_text())["metrics"]
    )


def test_supplied_panel_reuses_one_answer_for_two_fixed_opponents(
    tmp_path, monkeypatch, synthetic_arena_df
):
    data = normalize_battles(synthetic_arena_df, "ComparIA")
    key = data.instructions.index[0]
    panel = pd.DataFrame(
        {
            "instruction_index": [key, key],
            "model_b": ARENA_MODELS[:2],
            "completion_b": ["first", "second"],
        }
    ).rename_axis("instruction_index")
    cfg = _default_args(result_folder=str(tmp_path), store_root=str(tmp_path / "cache"))

    def evaluate(folder):
        manifest = pairwise_runner.run_evaluation(
            cfg, data=data, opponent_panel=panel, output_dir=folder
        )
        return json.loads(manifest.read_text())

    live = tmp_path / "live"
    saved = evaluate(live)
    outcomes = pd.read_parquet(live / saved["artifacts"]["metric_data"])
    assert set(saved["metrics"]["bradley_terry"]["reference_ratings"]) == set(
        ARENA_MODELS
    )
    assert len(pd.read_parquet(live / "answers.parquet")) == 1
    assert (
        outcomes.model_b.tolist()
        == outcomes.baseline.tolist()
        == panel.model_b.tolist()
    )
    assert outcomes.model_a.tolist() == [cfg.model.name] * 2
    assert (
        outcomes.instruction.tolist() == [data.instructions.loc[key, "instruction"]] * 2
    )
    monkeypatch.setattr(
        models, "make_model", lambda *a, **kw: pytest.fail("cache miss")
    )
    replay = evaluate(tmp_path / "replay")
    assert replay["metrics"] == saved["metrics"]
    pd.testing.assert_frame_equal(
        outcomes,
        pd.read_parquet(tmp_path / "replay" / replay["artifacts"]["metric_data"]),
    )
