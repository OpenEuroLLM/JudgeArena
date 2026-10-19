import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import judgearena.benchmarks.elo.calibration as elo_calibration
import judgearena.benchmarks.elo.runner as estimate_elo_ratings
from judgearena.benchmarks.elo.rating import (
    arena_anchor_battles,
    fit_bradley_terry,
    prefs_to_battle_results,
    winner_to_pref,
)
from judgearena.benchmarks.elo.runner import run_elo
from judgearena.benchmarks.elo.scoring import BradleyTerryMetric
from judgearena.cache.sqlite import JudgementCache
from judgearena.config import RunConfig, load_config
from judgearena.evaluate import judge_and_parse_prefs
from judgearena.models import DummyModel, make_model
from judgearena.prompts.parsing import JudgeParser, ParsedPreference
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import MetricSpec

N_BATTLES = 30
ARENA_MODELS = ["arena_model_alpha", "arena_model_beta", "arena_model_gamma"]


def _make_conversation(content_user: str, content_assistant: str) -> list[dict]:
    return [
        {"role": "user", "content": content_user},
        {"role": "assistant", "content": content_assistant},
    ]


def _arena_df(n_battles: int) -> pd.DataFrame:
    """Synthetic arena DataFrame matching the schema produced by load_arena_dataframe."""
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
        estimate_elo_ratings,
        "load_battles",
        lambda _task: synthetic_arena_df,
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

    monkeypatch.setattr(estimate_elo_ratings, "generate_instructions", mock_generate)


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
    battle_thinking_token_budget = kwargs.pop("battle_thinking_token_budget", None)
    store_root = kwargs.pop("store_root", None)
    assert not kwargs, f"unexpected kwargs: {kwargs}"
    judge: dict[str, object] = {
        "model": judge_model,
        "swap_mode": swap_mode,
        "strip_thinking_before_judging": strip_thinking_before_judging,
    }
    if battle_thinking_token_budget is not None:
        judge["battle_thinking_token_budget"] = battle_thinking_token_budget
    return RunConfig(
        task=task,
        model={"name": model},
        judge=judge,
        generation={"n_instructions": n_instructions},
        elo={
            "n_bootstraps": n_bootstraps,
            "languages": languages,
            "calibrate_temperature": calibrate_temperature,
        },
        run={"result_folder": result_folder, "store_root": store_root},
    )


def test_missing_preference_remains_missing_in_hard_battles():
    battles = prefs_to_battle_results([float("nan")], [True], ["opponent"], "candidate")

    assert pd.isna(battles.loc[0, "pref"])
    assert pd.isna(battles.loc[0, "pref_hard"])
    assert battles.loc[0, "winner"] is None


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
            "source": ["human", "human", "llm-judge", "llm-judge"],
            "evaluation_model": [None, None, "candidate", "candidate"],
        }
    )
    result = BradleyTerryMetric(n_bootstraps=3, baseline_model="anchor-b").calculate(
        battles, rng=np.random.default_rng(17)
    )
    assert result["ratings"]["anchor-b"] == pytest.approx(1000)
    assert set(result["human_ratings"]) == {"anchor-a", "anchor-b"}
    assert len(result["bootstrap_ratings"]) == result["n_bootstraps"] == 3
    assert result["evaluation_model"] == "candidate"
    assert result["battle_counts"]["candidate"] == 2
    assert {entry["model"] for entry in result["rating_entries"]} == {
        "anchor-a",
        "anchor-b",
        "candidate",
    }


# --- fit_bradley_terry unit tests ---


def _records_with_pref(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["pref"] = df["winner"].map(winner_to_pref)
    return df


def test_bradley_terry_clear_winner():
    """Model A always beats B → A gets a higher ELO."""
    records = [{"model_a": "A", "model_b": "B", "winner": "model_a"}] * 10 + [
        {"model_a": "B", "model_b": "A", "winner": "model_b"}
    ] * 10
    ratings = fit_bradley_terry(_records_with_pref(records))
    assert ratings["A"] > ratings["B"]


def test_bradley_terry_all_ties():
    """All ties → ratings should be equal."""
    records = [{"model_a": "A", "model_b": "B", "winner": "tie"}] * 20
    ratings = fit_bradley_terry(_records_with_pref(records))
    assert abs(ratings["A"] - ratings["B"]) < 1.0


def test_bradley_terry_baseline():
    """Baseline model is anchored at baseline_rating."""
    records = [{"model_a": "A", "model_b": "B", "winner": "model_a"}] * 10
    ratings = fit_bradley_terry(
        _records_with_pref(records),
        baseline_model="B",
        baseline_rating=1000,
    )
    assert ratings["B"] == pytest.approx(1000.0)
    assert ratings["A"] > 1000.0


def test_bradley_terry_soft_matches_hard():
    """Soft prefs ∈ {0, 0.5, 1} must give the same fit as hard winner labels."""
    records = (
        [{"model_a": "A", "model_b": "B", "winner": "model_a"}] * 7
        + [{"model_a": "A", "model_b": "B", "winner": "model_b"}] * 3
        + [{"model_a": "A", "model_b": "B", "winner": "tie"}] * 2
    )
    df = _records_with_pref(records)
    hard = fit_bradley_terry(df, pref_col="pref")
    # Passing the same column twice (continuous == quantised here) must match.
    df["pref_soft"] = df["pref"].astype(float)
    soft = fit_bradley_terry(df, pref_col="pref_soft")
    assert hard["A"] == pytest.approx(soft["A"], abs=1e-3)
    assert hard["B"] == pytest.approx(soft["B"], abs=1e-3)


# --- run_elo() integration tests ---


def run_elo_with_task(cfg: RunConfig) -> dict:
    return run_elo(cfg, get_packaged_task(cfg.task))


def _pairwise_metric(run_result: dict) -> dict:
    return run_result["metrics"]["pairwise_win_rate"]


def _rating_metric(run_result: dict) -> dict:
    return run_result["metrics"]["bradley_terry"]


def _num_pairwise_rows(run_result: dict) -> int:
    metric = _pairwise_metric(run_result)
    return (
        metric["num_wins"]
        + metric["num_losses"]
        + metric["num_ties"]
        + metric["num_missing"]
    )


def test_run_elo_returns_metrics(tmp_path):
    result = run_elo_with_task(_default_args(result_folder=str(tmp_path)))

    assert set(result) >= {
        "arena",
        "judge_model",
        "metrics",
        "model_name",
        "num_battles",
        "sampling_metadata",
        "result_path",
    }
    assert set(result["metrics"]) == {"pairwise_win_rate", "bradley_terry"}
    assert set(_pairwise_metric(result)) >= {
        "num_wins",
        "num_losses",
        "num_ties",
        "winrate",
    }
    assert "bootstrap_ratings" in _rating_metric(result)
    assert "rating_entries" in _rating_metric(result)
    assert "winrate" not in result
    assert "bootstrap_ratings" not in result


@pytest.mark.parametrize(
    ("runtime", "n_bootstraps", "method", "anchor"),
    [
        (None, 2, "ELO", ARENA_MODELS[0]),
        (
            {"n_bootstraps": 0, "baseline_model": None, "soft_elo": True},
            0,
            "Soft-ELO",
            None,
        ),
    ],
)
def test_run_elo_uses_actual_task_parameters_and_saves_resolved_settings(
    monkeypatch, tmp_path, runtime, n_bootstraps, method, anchor
):
    cfg = RunConfig(
        task="elo-comparia",
        model={"name": "Dummy/my model"},
        judge={"model": "Dummy/score A: 6 score B: 8"},
        generation={"n_instructions": 10},
        elo=runtime,
        run={"result_folder": str(tmp_path)},
    )
    task = get_packaged_task(cfg.task)
    scoring = task.spec.protocol.scoring.model_copy(
        update={
            "metrics": (
                MetricSpec(
                    metric="bradley_terry",
                    parameters={
                        "n_bootstraps": 2,
                        "baseline_model": ARENA_MODELS[0],
                        "soft": False,
                    },
                ),
            ),
            "default_temperature": 0.4,
        }
    )
    protocol = task.spec.protocol.model_copy(update={"scoring": scoring})
    task = replace(task, spec=task.spec.model_copy(update={"protocol": protocol}))
    calibration_args = {}
    calibrate = estimate_elo_ratings.calibrate_pairscore_temperature

    def capture_calibration(*args, **kwargs):
        calibration_args.update(kwargs)
        return calibrate(*args, **kwargs)

    monkeypatch.setattr(
        estimate_elo_ratings, "calibrate_pairscore_temperature", capture_calibration
    )

    result = run_elo(cfg, task)

    metric = _rating_metric(result)
    assert metric["method"] == method
    assert metric["n_bootstraps"] == n_bootstraps
    assert len(metric["bootstrap_ratings"]) == n_bootstraps
    if anchor is not None:
        assert metric["ratings"][anchor] == pytest.approx(1000)
    else:
        assert np.mean(list(metric["ratings"].values())) == pytest.approx(1000)
    assert cfg.elo.baseline_model == anchor
    assert calibration_args["soft_elo"] is (method == "Soft-ELO")
    assert calibration_args["default_temperature"] == 0.4
    saved = load_config(next(tmp_path.rglob("config.yaml")))
    assert saved.elo == cfg.elo


def test_run_elo_without_bradley_terry_skips_rating_artifacts(tmp_path):
    cfg = _default_args(result_folder=str(tmp_path))
    task = get_packaged_task(cfg.task)
    scoring = task.spec.protocol.scoring.model_copy(
        update={"metrics": (MetricSpec(metric="pairwise_win_rate"),)}
    )
    protocol = task.spec.protocol.model_copy(update={"scoring": scoring})
    task = replace(task, spec=task.spec.model_copy(update={"protocol": protocol}))

    result = run_elo(cfg, task)

    assert set(result["metrics"]) == {"pairwise_win_rate"}
    assert not list(tmp_path.rglob("bootstrap_ratings.csv"))
    assert not list(tmp_path.rglob("elo_ratings.json"))


def test_run_elo_winrate_in_valid_range(tmp_path):
    result = run_elo_with_task(_default_args(result_folder=str(tmp_path)))
    assert 0.0 <= _pairwise_metric(result)["winrate"] <= 1.0


def test_run_elo_winrate_depends_on_judge(tmp_path):
    """A judge biased toward one position should yield different winrates depending on direction."""
    # With seed=0 and n=10 our model is always placed in position B, so:
    # judge favouring B → all wins; judge favouring A → all losses
    result_wins = run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path), judge_model="Dummy/score A: 0 score B: 10"
        )
    )
    result_loses = run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path), judge_model="Dummy/score A: 10 score B: 0"
        )
    )
    assert (
        _pairwise_metric(result_wins)["winrate"]
        > _pairwise_metric(result_loses)["winrate"]
    )


def test_run_elo_language_filter_reduces_battles(tmp_path):
    """Filtering to a single language should use fewer battles than no filter."""
    result_all = run_elo_with_task(
        _default_args(result_folder=str(tmp_path), n_instructions=None)
    )
    result_en = run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path), n_instructions=None, languages=["en"]
        )
    )
    total_all = _num_pairwise_rows(result_all)
    total_en = _num_pairwise_rows(result_en)
    assert total_en < total_all


def test_run_elo_model_in_bootstrap_ratings(tmp_path):
    """Our model should appear in the bootstrap ELO leaderboard."""
    result = run_elo_with_task(_default_args(result_folder=str(tmp_path)))
    model_name = result["model_name"]
    assert all(
        model_name in ratings for ratings in _rating_metric(result)["bootstrap_ratings"]
    )


def test_run_elo_n_instructions_limits_battles(tmp_path):
    """n_instructions caps the number of judged battles."""
    result_5 = run_elo_with_task(
        _default_args(result_folder=str(tmp_path), n_instructions=5)
    )
    result_10 = run_elo_with_task(
        _default_args(result_folder=str(tmp_path), n_instructions=10)
    )
    total_5 = _num_pairwise_rows(result_5)
    total_10 = _num_pairwise_rows(result_10)
    assert total_5 == 5
    assert total_10 == 10


@pytest.mark.parametrize(
    ("swap_mode", "strip_thinking"),
    [("fixed", False), ("random", True), ("both", True)],
)
def test_run_elo_aligns_shuffled_completions_with_judged_battles(
    monkeypatch, tmp_path, swap_mode, strip_thinking
):
    def generate(instructions, **_kwargs):
        return pd.DataFrame(
            {
                "instruction_index": instructions.index,
                "completion": [
                    f"<think>private {i}</think>Answer {i}"
                    for i in range(len(instructions))
                ],
            }
        ).iloc[::-1]

    monkeypatch.setattr(estimate_elo_ratings, "generate_instructions", generate)
    cfg = _default_args(
        result_folder=str(tmp_path),
        store_root=str(tmp_path / "cache"),
        n_instructions=3,
        n_bootstraps=0,
        swap_mode=swap_mode,
        strip_thinking_before_judging=strip_thinking,
    )
    cfg.run.seed = 42
    run_elo_with_task(cfg)

    with JudgementCache(next((tmp_path / "cache").rglob("judgements.db"))) as cache:
        cached = cache.query()
    assert len(cached) == (6 if swap_mode == "both" else 3)
    for row in cached.itertuples():
        i = int(row.instruction_id.rsplit("q", 1)[1])
        assert f"Answer {i}" in row.judge_input
        assert ("private" in row.judge_input) is not strip_thinking
        assert row.orientation == (
            "direct" if row.model_a == cfg.model.name else "reversed"
        )

    battles = pd.read_parquet(next(tmp_path.rglob("battles.parquet")))
    repeats = 2 if swap_mode == "both" else 1
    assert battles["question_id"].tolist() == ["q0", "q1", "q2"] * repeats
    assert (battles["model_a"] == cfg.model.name).tolist() == [
        True,
        True,
        False,
    ] * repeats
    direct_pref = 1 / (1 + math.exp(-cfg.elo.soft_elo_temperature * 10))
    expected = [direct_pref] * 3
    if swap_mode == "both":
        expected += [1 - direct_pref] * 3
    assert battles["pref"].tolist() == pytest.approx(expected)


def test_run_elo_keeps_battles_with_repeated_question_ids(tmp_path, synthetic_arena_df):
    synthetic_arena_df.loc[1, "question_id"] = "q0"
    run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path),
            n_instructions=3,
            n_bootstraps=0,
            swap_mode="both",
        )
    )
    battles = pd.read_parquet(next(tmp_path.rglob("battles.parquet")))
    assert battles["question_id"].tolist() == ["q0", "q0", "q2"] * 2


def _spy_generate_capturing(captured):
    def spy_generate(instructions, model, **kwargs):
        captured["gen_kwargs"] = kwargs
        return pd.DataFrame(
            {
                "completion": [f"c{i}" for i in range(len(instructions))],
                "instruction_index": instructions.index,
            }
        )

    return spy_generate


def test_run_elo_thinking_budget_injected_for_thinking_model(monkeypatch, tmp_path):
    """battle_thinking_token_budget must reach generation for VLLM thinking models."""
    captured = {}
    monkeypatch.setattr(
        estimate_elo_ratings, "generate_instructions", _spy_generate_capturing(captured)
    )
    run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path),
            model="VLLM/Qwen/Qwen3.5-9B",
            battle_thinking_token_budget=128,
        )
    )
    assert captured["gen_kwargs"].get("thinking_token_budget") == 128


def test_run_elo_thinking_budget_capped_by_max_out_tokens(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        estimate_elo_ratings, "generate_instructions", _spy_generate_capturing(captured)
    )
    cfg = _default_args(
        result_folder=str(tmp_path),
        model="VLLM/Qwen/Qwen3.5-9B",
        battle_thinking_token_budget=10**9,
    )
    run_elo_with_task(cfg)
    assert (
        captured["gen_kwargs"].get("thinking_token_budget") == cfg.model.max_out_tokens
    )


def test_run_elo_thinking_budget_absent_for_nonthinking_model(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        estimate_elo_ratings, "generate_instructions", _spy_generate_capturing(captured)
    )
    run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path),
            model="Dummy/my model",
            battle_thinking_token_budget=128,
        )
    )
    assert "thinking_token_budget" not in captured["gen_kwargs"]


def test_judge_and_parse_prefs_retains_structured_result():
    judge = make_model("Dummy/Score_A: 6\nScore_B: 8")

    annotations, _, prefs = judge_and_parse_prefs(
        judge_chat_model=judge,
        instructions=["Q"],
        completions_A=["A"],
        completions_B=["B"],
    )

    assert prefs.tolist() == pytest.approx([0.6456563062257954])
    assert annotations[0].parsed is not None
    assert annotations[0].parsed.scores == {"A": 6.0, "B": 8.0}


def test_judge_and_parse_prefs_none_prefs_swap_mode_both():
    """swap_mode='both' must not raise when judge output is unparseable (None prefs).

    Regression test: previously '1 - prefs_reversed' raised TypeError when
    prefs_reversed contained None values from an unparseable judge completion.
    """
    judge = make_model("Dummy/no scores here at all")
    instructions = ["Q1", "Q2", "Q3"]
    completions_A = ["A1", "A2", "A3"]
    completions_B = ["B1", "B2", "B3"]

    _, _, prefs = judge_and_parse_prefs(
        judge_chat_model=judge,
        instructions=instructions,
        completions_A=completions_A,
        completions_B=completions_B,
        swap_mode="both",
    )
    # All prefs should be NaN (unparseable → nan), not raise
    assert all(math.isnan(p) for p in prefs)


def test_arena_anchor_battles_filters_and_preserves_index():
    # Anchors are rebuilt on recompute, so this primitive must drop under-
    # represented models (< 500 battles), keep provenance, and preserve the
    # arena row labels (calibration looks up conversations via df_arena_all.loc[i]).
    n = 500
    df_all = pd.DataFrame(
        {
            "model_a": ["x"] * n + ["rare"],
            "model_b": ["y"] * n + ["x"],
            "winner": ["model_a", "model_b"] * (n // 2) + ["model_a"],
            "conversation_a": [["q"]] * (n + 1),  # extra column must be ignored
        },
        index=range(1000, 1000 + n + 1),
    )
    out = arena_anchor_battles(df_all)

    # x, y have >= 500 battles -> kept; 'rare' (1 battle) -> its row dropped
    assert set(out["model_a"]) | set(out["model_b"]) == {"x", "y"}
    assert list(out.index) == list(range(1000, 1000 + n))  # labels preserved, rare gone
    assert (out["source"] == "human").all()
    assert out.loc[1000, "pref"] == 0.0 and out.loc[1001, "pref"] == 1.0


def test_elo_language_variant_resolves_and_filters(tmp_path):
    variant = get_packaged_task("elo-lmarena-140k-en")
    assert variant is not None
    assert variant.selection is not None
    assert variant.selection.values == ("en",)

    result_en = run_elo(
        _default_args(
            result_folder=str(tmp_path), task="elo-lmarena-140k-en", n_instructions=None
        ),
        variant,
    )
    result_all = run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path), task="elo-lmarena-140k", n_instructions=None
        )
    )
    total_en = _num_pairwise_rows(result_en)
    total_all = _num_pairwise_rows(result_all)
    assert 0 < total_en < total_all


@pytest.mark.parametrize("swap_mode", ["fixed", "both"])
def test_run_elo_temperature_calibration_builds_judge(monkeypatch, tmp_path, swap_mode):
    """Regression: the calibration path constructs its own judge model and once
    crashed on a duplicate max_tokens kwarg; nothing else exercises it. The
    MLE fit itself is mocked."""
    captured = {}

    def fake_calibrate(delta_s, y):
        captured["n_pairs"] = len(delta_s)
        return 0.42

    monkeypatch.setattr(elo_calibration, "fit_temperature", fake_calibrate)
    # Anchor battles require models with >= 500 appearances; the default
    # 30-battle fixture leaves the calibration pool empty.
    arena = _arena_df(900)
    monkeypatch.setattr(estimate_elo_ratings, "load_battles", lambda _task: arena)

    store_root = tmp_path / "cache"
    result = run_elo_with_task(
        _default_args(
            result_folder=str(tmp_path),
            calibrate_temperature=True,
            swap_mode=swap_mode,
            store_root=str(store_root),
        )
    )

    assert captured["n_pairs"] == (arena["winner"] != "tie").sum()
    assert 0.0 <= _pairwise_metric(result)["winrate"] <= 1.0
    battles = pd.read_parquet(next(tmp_path.rglob("battles.parquet")))
    pref = 1 / (1 + math.exp(-0.42 * 10))
    expected = [pref] * 10 + ([1 - pref] * 10 if swap_mode == "both" else [])
    assert battles["pref"].tolist() == pytest.approx(expected)
    judgement_db = next((store_root / "judgements").rglob("judgements.db"))
    with JudgementCache(judgement_db) as cache:
        assert cache.query()["instruction_id"].str.startswith("ComparIA:").all()


def test_empty_calibration_keeps_configured_temperature(tmp_path):
    run_elo_with_task(
        _default_args(result_folder=str(tmp_path), calibrate_temperature=True)
    )
    battles = pd.read_parquet(next(tmp_path.rglob("battles.parquet")))
    assert battles["pref"].tolist() == pytest.approx([0.9525741268224334] * 10)


def test_extract_turn_text_tolerates_moderated_turns():
    from judgearena.arenas_utils import extract_turn_text

    assert extract_turn_text({"content": None}) == ""
    assert extract_turn_text({"content": "plain"}) == "plain"
    assert (
        extract_turn_text(
            {"content": [{"type": "text", "text": None}, {"type": "image"}, None]}
        )
        == ""
    )


def test_run_elo_preserves_soft_preferences_from_non_pairscore_parser(
    tmp_path, monkeypatch
):
    class SoftParser(JudgeParser):
        name = "soft-parser"

        def parse_result(self, judge_completion, **kwargs):
            return ParsedPreference(0.75)

    monkeypatch.setattr(
        estimate_elo_ratings,
        "resolve_run_judge_prompt",
        lambda *_args, **_kwargs: SimpleNamespace(
            parser=SoftParser(),
            preset_name="soft-parser",
            system_prompt="system",
            user_prompt_template="{user_prompt} {completion_A} {completion_B}",
        ),
    )
    captured_prefs = []
    convert = estimate_elo_ratings.prefs_to_battle_results

    def capture_prefs(prefs, *args, **kwargs):
        captured_prefs.extend(prefs)
        return convert(prefs, *args, **kwargs)

    monkeypatch.setattr(estimate_elo_ratings, "prefs_to_battle_results", capture_prefs)
    run_elo_with_task(_default_args(result_folder=str(tmp_path)))
    assert captured_prefs and set(captured_prefs) == {0.75}


def test_run_elo_temperature_preserves_selected_parser(monkeypatch, tmp_path):
    from judgearena.prompts.parsing import JUDGE_PARSERS

    parser = JUDGE_PARSERS["meta-eval-score"]
    monkeypatch.setattr(
        DummyModel,
        "batch",
        lambda self, inputs, **kwargs: [
            "Score_A: 6\nScore_B: 8",
            "Score_A: 6\nScore_B: 8.5",
        ],
    )
    cfg = _default_args(result_folder=str(tmp_path), n_instructions=2, swap_mode="both")
    cfg.elo.n_bootstraps = 0
    cfg.judge.prompt_preset = "meta-eval-pair-score"
    cfg.elo.soft_elo_temperature = 0.8

    result = run_elo_with_task(cfg)

    battles = pd.read_parquet(next(tmp_path.rglob("battles.parquet")))
    assert battles["pref"].tolist() == pytest.approx(
        [0.8320183851339245, float("nan"), 0.1679816148660755, float("nan")],
        nan_ok=True,
    )
    assert battles.loc[battles["pref"].isna(), "pref_hard"].isna().all()
    assert _pairwise_metric(result)["num_missing"] == 2
    assert parser.temperature == 0.5
