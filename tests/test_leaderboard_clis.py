"""Tests for the leaderboard CLIs (offline; judge/model/generation mocked)."""

from __future__ import annotations

import json

import pandas as pd

from judgearena.estimate_elo_ratings import _slugify
from judgearena.leaderboard.board import build_board, main_board, render_board
from judgearena.leaderboard.panel import PANEL_BATTLE_COLUMNS, Panel, save_panel
from judgearena.leaderboard.record import ResultRecord


def _save_tiny_panel(directory):
    meta = {
        "panel_version": "pv1",
        "judge_model": "OpenRouter/judge",
        "baseline_model": None,
        "generation_params": {"max_out_tokens": 256, "truncate_all_input_chars": 1000},
        "scorer": {"method": "soft", "temperature": 0.3, "calibrated": False},
    }
    battles = pd.DataFrame(columns=list(PANEL_BATTLE_COLUMNS))
    return save_panel(Panel(meta=meta, battles=battles), directory)


def test_main_submit_uses_panel_judge_and_writes_record(tmp_path, monkeypatch):
    import judgearena.leaderboard.submit as sub
    from judgearena.evaluate import PairScore

    panel_dir = _save_tiny_panel(tmp_path / "panel")
    captured = {}

    monkeypatch.setattr(sub, "_download_panel", lambda repo, version: panel_dir)
    monkeypatch.setattr(sub, "_resolve_panel_version", lambda files: "pv1")

    # Realistic battles: 2 rows over 1 instruction (will dedup to 1 new_completion row).
    _pool_battles = pd.DataFrame({
        "instruction": ["q1", "q1"],
        "lang": ["en", "en"],
        "model_a": ["anchor1", "anchor2"],
        "model_b": ["anchor2", "anchor1"],
    })

    def _fake_load_pool(_dir):
        from judgearena.leaderboard.panel import Panel
        return (
            Panel(
                meta={
                    "panel_version": "pv1", "panel_hash": "ph",
                    "judge_model": "OpenRouter/judge", "baseline_model": None,
                    "generation_params": {"max_out_tokens": 256, "truncate_all_input_chars": 1000},
                    "scorer": {"method": "soft", "temperature": 0.3, "calibrated": False},
                    "pool_models": [], "arena": "test-arena",
                },
                battles=_pool_battles,
            ),
            pd.DataFrame(columns=["model", "instruction", "lang", "completion"]),
        )

    monkeypatch.setattr(sub, "load_pool", _fake_load_pool)

    # generate_panel_completions must return a list of len(panel.battles) == 2.
    monkeypatch.setattr(sub, "generate_panel_completions",
                        lambda panel, model, **kw: ["comp1", "comp1"])

    # Regression guard: capture new_completions and scorer passed to sample_pool_battles
    # and judge_pool_battles to verify their types.
    _sampled_battles = pd.DataFrame({
        "model_a": [], "model_b": [], "instruction": [], "lang": [],
    })

    def _capturing_sample(new_model, new_completions, panel, pool_completions, **kw):
        captured["new_completions"] = new_completions
        return _sampled_battles

    monkeypatch.setattr(sub, "sample_pool_battles", _capturing_sample)

    def _capturing_judge(specs, new_model, *, judge_cfg, scorer, arena=None):
        captured["scorer"] = scorer
        return pd.DataFrame({
            "model_a": [], "model_b": [], "instruction": [], "lang": [],
            "judge_pref": [], "judge_pref_hard": [], "position": [], "opponent": [],
        })

    monkeypatch.setattr(sub, "judge_pool_battles", _capturing_judge)

    def fake_place(panel, new_model, new_battles, *, n_bootstraps, seed):
        captured["judge"] = panel.meta["judge_model"]
        captured["gen"] = panel.meta["generation_params"]
        captured["n_bootstraps"] = n_bootstraps
        return ResultRecord(
            model=new_model, panel_version=panel.meta["panel_version"],
            panel_hash=panel.meta["panel_hash"], judge_model=panel.meta["judge_model"],
            elo_overall=1010.0, elo_std=5.0, elo_ci=[1000.0, 1020.0],
            elo_per_lang={}, winrate_overall=0.5, winrate_per_lang={},
            n_battles=2, n_battles_per_lang={}, kappa_per_lang={},
            mae_vs_human=float("nan"), scorer=panel.meta["scorer"],
            generation_params={}, seed=seed,
        )

    monkeypatch.setattr(sub, "place_against_pool", fake_place)

    model = "VLLM/openeurollm/OLMo-3-7B-Dolci-Translated-A-75EN"
    out = sub.main_submit([
        "--repo", "u/lb", "--model", model,
        "--out", str(tmp_path / "results"), "--n-bootstraps", "7",
        "--panel-version", "pv1", "--dry-run",
    ])

    assert out == tmp_path / "results" / "pv1" / _slugify(model)
    assert (out / "result.json").exists()
    payload = json.loads((out / "result.json").read_text())
    assert payload["elo_overall"] == 1010.0
    assert captured["judge"] == "OpenRouter/judge"        # panel's frozen judge
    assert captured["gen"]["max_out_tokens"] == 256       # panel's frozen generation
    assert captured["n_bootstraps"] == 7

    # Regression guard: new_completions must be a DataFrame with the right columns.
    nc = captured["new_completions"]
    assert isinstance(nc, pd.DataFrame), "new_completions must be a DataFrame, not a list"
    assert list(nc.columns) == ["model", "instruction", "lang", "completion"]
    # Regression guard: scorer must be a PairScore instance.
    assert isinstance(captured["scorer"], PairScore), \
        "scorer passed to judge_pool_battles must be a PairScore instance"


def _board_panel():
    battles = pd.DataFrame(
        {
            "battle_id": ["b1", "b2", "b3"],
            "lang": ["en", "en", "fr"],
            "model_a": ["strong", "strong", "strong"],
            "model_b": ["weak", "weak", "weak"],
            "instruction": ["q1", "q2", "q3"],
            "completion_a": ["a", "a", "a"],
            "completion_b": ["b", "b", "b"],
            "human_winner": ["model_a", "model_a", "model_a"],
            "judge_pref": [0.0, 0.0, 0.0],          # strong (A) always wins
            "judge_pref_hard": [0.0, 0.0, 0.0],
            "challenger_opponent": ["strong", "strong", "strong"],
            "challenger_position": ["A", "A", "A"],
        }
    )
    meta = {
        "panel_version": "pv1", "panel_hash": "H", "baseline_model": "weak",
        "scorer": {"method": "soft", "temperature": 0.3, "calibrated": False},
    }
    return Panel(meta=meta, battles=battles)


def _record(model, elo, panel_hash="H"):
    return {
        "model": model, "panel_hash": panel_hash, "elo_overall": elo,
        "elo_ci": [elo - 10, elo + 10], "n_battles": 3,
        "elo_per_lang": {"en": elo + 1, "fr": elo - 1},
    }


def test_build_board_merges_anchors_and_submissions_sorted():
    panel = _board_panel()
    records = [_record("cand-mid", 1050.0), _record("cand-other-panel", 9999.0, panel_hash="X")]
    board = build_board(panel, records)
    # anchors present, matching submission present, mismatched submission excluded
    models = list(board["model"])
    assert "strong" in models and "weak" in models and "cand-mid" in models
    assert "cand-other-panel" not in models
    # sorted by elo descending with ranks 1..n
    assert list(board["rank"]) == list(range(1, len(board) + 1))
    assert board["elo"].is_monotonic_decreasing
    # the submission flag is set only on the submitted model
    assert bool(board.loc[board["model"] == "cand-mid", "is_submission"].iloc[0]) is True
    assert bool(board.loc[board["model"] == "strong", "is_submission"].iloc[0]) is False


def test_build_board_lang_uses_per_language_values():
    panel = _board_panel()
    board = build_board(panel, [_record("cand-mid", 1050.0)], lang="en")
    assert board.loc[board["model"] == "cand-mid", "elo"].iloc[0] == 1051.0  # elo_per_lang["en"]


def test_render_board_formats():
    panel = _board_panel()
    board = build_board(panel, [_record("cand-mid", 1050.0)])
    assert "cand-mid" in render_board(board, "table")
    assert "| Rank |" in render_board(board, "markdown")
    assert "rank,model,elo" in render_board(board, "csv").splitlines()[0]


def test_main_board_reads_records_and_prints(tmp_path, capsys):
    # a panel on disk (real panel_hash) + one matching record under results/<pv>/<slug>/
    panel_dir = _save_tiny_panel(tmp_path / "panel")  # empty-battles panel is fine here
    from judgearena.leaderboard.panel import load_panel
    ph = load_panel(panel_dir).meta["panel_hash"]

    rec_dir = tmp_path / "results" / "pv1" / "cand-mid"
    rec_dir.mkdir(parents=True)
    (rec_dir / "result.json").write_text(json.dumps({
        "model": "cand-mid", "panel_hash": ph, "elo_overall": 1050.0,
        "elo_ci": [1040.0, 1060.0], "n_battles": 3, "elo_per_lang": {},
    }))

    main_board([
        "--panel-dir", str(panel_dir),
        "--results-dir", str(tmp_path / "results"),
        "--format", "markdown",
    ])
    out = capsys.readouterr().out
    assert "cand-mid" in out
    assert "| Rank |" in out


def test_main_submit_tag_suffixes_dir_and_sets_record(tmp_path, monkeypatch):
    import judgearena.leaderboard.submit as sub
    from judgearena.estimate_elo_ratings import _slugify

    monkeypatch.setattr(sub, "_download_panel", lambda repo, version: tmp_path / "panel" / "pv1")
    monkeypatch.setattr(sub, "_resolve_panel_version", lambda files: "pv1")

    # Realistic battles: 2 rows, 2 instructions (will produce 2 new_completion rows).
    _pool_battles = pd.DataFrame({
        "instruction": ["q1", "q2"],
        "lang": ["en", "fr"],
        "model_a": ["anchor1", "anchor1"],
        "model_b": ["anchor2", "anchor2"],
    })

    def _fake_load_pool(_dir):
        from judgearena.leaderboard.panel import Panel
        return (
            Panel(
                meta={
                    "panel_version": "pv1", "panel_hash": "ph",
                    "judge_model": "j", "baseline_model": None,
                    "generation_params": {}, "scorer": {"temperature": 0.3}, "pool_models": [],
                },
                battles=_pool_battles,
            ),
            pd.DataFrame(columns=["model", "instruction", "lang", "completion"]),
        )

    monkeypatch.setattr(sub, "load_pool", _fake_load_pool)
    # generate_panel_completions must return a list of len(panel.battles) == 2.
    monkeypatch.setattr(sub, "generate_panel_completions", lambda *a, **k: ["c1", "c2"])
    monkeypatch.setattr(sub, "sample_pool_battles", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sub, "judge_pool_battles", lambda *a, **k: pd.DataFrame({
        "model_a": [], "model_b": [], "instruction": [], "lang": [],
        "judge_pref": [], "judge_pref_hard": [], "position": [], "opponent": [],
    }))

    saved = {}

    def fake_place(panel, new_model, new_battles, *, n_bootstraps, seed):
        from judgearena.leaderboard.record import ResultRecord
        rec = ResultRecord(
            model=new_model, panel_version=panel.meta["panel_version"],
            panel_hash=panel.meta["panel_hash"], judge_model=panel.meta["judge_model"],
            elo_overall=1000.0, elo_std=0.0, elo_ci=[1000.0, 1000.0],
            elo_per_lang={}, winrate_overall=0.5, winrate_per_lang={},
            n_battles=1, n_battles_per_lang={}, kappa_per_lang={},
            mae_vs_human=float("nan"), scorer={}, generation_params={}, seed=seed,
        )
        saved["rec"] = rec
        return rec

    monkeypatch.setattr(sub, "place_against_pool", fake_place)

    model = "OpenRouter/deepseek/deepseek-v3.2"
    out = sub.main_submit([
        "--repo", "u/lb", "--model", model,
        "--out", str(tmp_path / "results"), "--tag", "seed-1",
        "--panel-version", "pv1", "--dry-run",
    ])
    assert out.name == f"{_slugify(model)}__{_slugify('seed-1')}"
    assert (out / "result.json").exists()
    assert saved["rec"].tag == "seed-1"


def test_build_board_distinguishes_same_model_by_tag():
    from judgearena.leaderboard.board import build_board
    panel = _board_panel()
    ph = panel.meta["panel_hash"]
    rec_a = {"model": "deepseek", "tag": "seed-1", "panel_hash": ph,
             "elo_overall": 1010.0, "elo_ci": [1000.0, 1020.0], "n_battles": 3, "elo_per_lang": {}}
    rec_b = {"model": "deepseek", "tag": "seed-2", "panel_hash": ph,
             "elo_overall": 1005.0, "elo_ci": [995.0, 1015.0], "n_battles": 3, "elo_per_lang": {}}
    board = build_board(panel, [rec_a, rec_b])
    labels = list(board["model"])
    assert "deepseek #seed-1" in labels
    assert "deepseek #seed-2" in labels


def test_curate_writes_anchor_caches(tmp_path, monkeypatch):
    # Build a tiny panel and drive only the cache-writing + save path.
    import json

    import pandas as pd

    from judgearena.leaderboard.anchors import save_anchor_caches
    from judgearena.leaderboard.panel import Panel, panel_hash, save_panel

    battles = pd.DataFrame(
        {
            "battle_id": ["a", "b"],
            "lang": ["en", "en"],
            "model_a": ["m1", "m2"],
            "model_b": ["m2", "m1"],
            "instruction": ["q", "q"],
            "completion_a": ["x", "x"],
            "completion_b": ["y", "y"],
            "human_winner": ["model_a", "model_a"],
            "judge_pref": [0.2, 0.3],
            "judge_pref_hard": [0.0, 0.0],
            "challenger_opponent": ["m2", "m1"],
            "challenger_position": ["A", "A"],
        }
    )
    meta = {"panel_version": "v1", "panel_hash": panel_hash(battles),
            "baseline_model": "m1", "scorer": {"method": "soft"},
            "kappa_per_language": {"en": 0.7}}
    panel = Panel(meta=meta, battles=battles)
    out = tmp_path / "v1"
    save_panel(panel, out)
    save_anchor_caches(panel, out)
    for name in ("anchor_ratings.json", "calibration.json", "anchor_h2h.json"):
        assert (out / name).exists()
    assert "overall" in json.loads((out / "anchor_ratings.json").read_text())


def test_submit_resolve_latest_panel_version():
    from judgearena.leaderboard.submit import _resolve_panel_version

    files = [
        "panel/v1/panel.json",
        "panel/v2/panel.json",
        "panel/v10/panel.json",
        "records/v2/x/result.json",
        "README.md",
    ]
    assert _resolve_panel_version(files) == "v10"


def test_submit_dry_run_skips_upload(tmp_path, monkeypatch):
    import pandas as pd

    import judgearena.leaderboard.submit as sub
    from judgearena.leaderboard.record import ResultRecord

    # Stub the pool download + scoring so the test stays offline.
    monkeypatch.setattr(sub, "_download_panel", lambda repo, version: tmp_path / "panel" / "v1")
    monkeypatch.setattr(sub, "_resolve_panel_version", lambda files: "v1")

    # Realistic battles: 3 rows over 2 instructions.
    _pool_battles_dry = pd.DataFrame({
        "instruction": ["q1", "q1", "q2"],
        "lang": ["en", "en", "fr"],
        "model_a": ["anchor1", "anchor2", "anchor1"],
        "model_b": ["anchor2", "anchor1", "anchor2"],
    })

    def _fake_load_pool(_dir):
        from judgearena.leaderboard.panel import Panel
        return (
            Panel(
                meta={"panel_version": "v1", "panel_hash": "h", "judge_model": "j",
                      "generation_params": {}, "scorer": {"method": "soft", "temperature": 0.3},
                      "pool_models": []},
                battles=_pool_battles_dry,
            ),
            pd.DataFrame(columns=["model", "instruction", "lang", "completion"]),
        )

    monkeypatch.setattr(sub, "load_pool", _fake_load_pool)
    # generate_panel_completions must return a list of len(panel.battles) == 3.
    monkeypatch.setattr(sub, "generate_panel_completions", lambda *a, **k: ["c1", "c1", "c2"])
    monkeypatch.setattr(sub, "sample_pool_battles", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sub, "judge_pool_battles", lambda *a, **k: pd.DataFrame({
        "model_a": [], "model_b": [], "instruction": [], "lang": [],
        "judge_pref": [], "judge_pref_hard": [], "position": [], "opponent": [],
    }))

    def _fake_place(*a, **k):
        return ResultRecord(
            model="sub", panel_version="v1", panel_hash="h", judge_model="j",
            elo_overall=1.0, elo_std=0.0, elo_ci=[0.0, 2.0], elo_per_lang={},
            winrate_overall=0.5, winrate_per_lang={}, n_battles=0,
            n_battles_per_lang={}, kappa_per_lang={}, mae_vs_human=0.0,
            scorer={}, generation_params={}, seed=0,
        )

    monkeypatch.setattr(sub, "place_against_pool", _fake_place)

    called = {"upload": False}

    def _fake_upload(**kwargs):
        called["upload"] = True
        return "pr-url"

    monkeypatch.setattr(sub, "upload_folder", _fake_upload, raising=False)

    out = sub.main_submit(
        ["--repo", "u/lb", "--model", "sub", "--out", str(tmp_path / "results"),
         "--panel-version", "v1", "--dry-run"]
    )
    assert (out / "result.json").exists()
    assert called["upload"] is False


def _fake_pool_panel_and_dir(tmp_path):
    """Shared helper: write a minimal panel.json + battles.parquet for pool tests."""
    import json

    import pandas as pd

    fake_panel_dir = tmp_path / "pool" / "v1"
    fake_panel_dir.mkdir(parents=True)
    (fake_panel_dir / "panel.json").write_text(json.dumps({
        "panel_version": "v1", "panel_hash": "h", "judge_model": "j",
        "generation_params": {}, "scorer": {"method": "soft"},
        "pool_models": ["anchor1"],
    }))
    # completions.parquet must exist for load_pool
    pd.DataFrame(columns=["model", "instruction", "lang", "completion"]).to_parquet(
        fake_panel_dir / "completions.parquet", index=False
    )
    pd.DataFrame().to_parquet(fake_panel_dir / "battles.parquet", index=False)
    return fake_panel_dir


def test_submit_into_pool_extends_and_bumps_version(tmp_path, monkeypatch):
    """--into-pool: extend_pool + save_pool are called, version is bumped, no record PR."""
    import json

    import pandas as pd

    import judgearena.leaderboard.submit as sub

    fake_panel_dir = _fake_pool_panel_and_dir(tmp_path)

    monkeypatch.setattr(sub, "_download_panel", lambda repo, version: fake_panel_dir)
    monkeypatch.setattr(sub, "_resolve_panel_version", lambda files: "v1")

    # Realistic battles: 2 rows over 2 instructions.
    _into_pool_battles = pd.DataFrame({
        "instruction": ["q1", "q2"],
        "lang": ["en", "fr"],
        "model_a": ["anchor1", "anchor1"],
        "model_b": ["anchor2", "anchor2"],
    })

    def _fake_load_pool(_dir):
        from judgearena.leaderboard.panel import Panel
        meta = json.loads((fake_panel_dir / "panel.json").read_text())
        meta["scorer"] = {"method": "soft", "temperature": 0.3}
        return (
            Panel(meta=meta, battles=_into_pool_battles),
            pd.DataFrame(columns=["model", "instruction", "lang", "completion"]),
        )

    monkeypatch.setattr(sub, "load_pool", _fake_load_pool)
    # generate_panel_completions must return a list of len(panel.battles) == 2.
    monkeypatch.setattr(sub, "generate_panel_completions", lambda *a, **k: ["c1", "c2"])

    new_battles_df = pd.DataFrame({
        "model_a": ["new"], "model_b": ["anchor1"],
        "instruction": ["q1"], "lang": ["en"],
        "judge_pref": [0.3], "judge_pref_hard": [0.0],
        "position": ["A"], "opponent": ["anchor1"],
    })

    monkeypatch.setattr(sub, "sample_pool_battles", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sub, "judge_pool_battles", lambda *a, **k: new_battles_df)

    called = {}

    def _fake_extend_pool(panel, completions, new_model, new_completions, new_battles, *, bump_version):
        called["extend_pool"] = {"bump_version": bump_version}
        from judgearena.leaderboard.panel import Panel
        new_meta = dict(panel.meta)
        new_meta["panel_version"] = bump_version
        return Panel(meta=new_meta, battles=new_battles), completions

    monkeypatch.setattr(sub, "extend_pool", _fake_extend_pool)

    def _fake_save_pool(panel, completions, directory):
        called["save_pool"] = {"version": panel.meta["panel_version"], "directory": directory}
        return directory

    monkeypatch.setattr(sub, "save_pool", _fake_save_pool)

    def _fake_save_anchor_caches(panel, directory):
        called["save_anchor_caches"] = True

    monkeypatch.setattr(sub, "save_anchor_caches", _fake_save_anchor_caches)

    upload_calls = []

    def _fake_upload(**kwargs):
        upload_calls.append(kwargs)
        return "upload-url"

    monkeypatch.setattr(sub, "upload_folder", _fake_upload, raising=False)

    out = sub.main_submit([
        "--repo", "u/lb", "--model", "new_model",
        "--out", str(tmp_path / "results"),
        "--panel-version", "v1",
        "--into-pool",
        "--dry-run",
    ])

    # extend_pool must have been called with a bumped version
    assert "extend_pool" in called
    assert called["extend_pool"]["bump_version"] == "v2"

    # save_pool must have been called with the bumped version
    assert "save_pool" in called
    assert called["save_pool"]["version"] == "v2"

    # anchor caches must be recomputed
    assert called.get("save_anchor_caches") is True

    # dry-run: no upload
    assert upload_calls == []

    # returned path is the pool dir with the bumped version
    assert out.name == "v2"


def test_submit_place_against_pool_default(tmp_path, monkeypatch):
    """Default mode + --dry-run: place_against_pool is called, record is saved, no PR."""
    import json

    import pandas as pd

    import judgearena.leaderboard.submit as sub
    from judgearena.leaderboard.record import ResultRecord

    fake_panel_dir = _fake_pool_panel_and_dir(tmp_path)

    monkeypatch.setattr(sub, "_download_panel", lambda repo, version: fake_panel_dir)
    monkeypatch.setattr(sub, "_resolve_panel_version", lambda files: "v1")

    # Realistic battles: 2 rows over 2 instructions.
    _place_battles = pd.DataFrame({
        "instruction": ["q1", "q2"],
        "lang": ["en", "fr"],
        "model_a": ["anchor1", "anchor1"],
        "model_b": ["anchor2", "anchor2"],
    })

    def _fake_load_pool(_dir):
        from judgearena.leaderboard.panel import Panel
        meta = json.loads((fake_panel_dir / "panel.json").read_text())
        meta["scorer"] = {"method": "soft", "temperature": 0.3}
        return (
            Panel(meta=meta, battles=_place_battles),
            pd.DataFrame(columns=["model", "instruction", "lang", "completion"]),
        )

    monkeypatch.setattr(sub, "load_pool", _fake_load_pool)
    # generate_panel_completions must return a list of len(panel.battles) == 2.
    monkeypatch.setattr(sub, "generate_panel_completions", lambda *a, **k: ["c1", "c2"])

    new_battles_df = pd.DataFrame({
        "model_a": ["new_model"], "model_b": ["anchor1"],
        "instruction": ["q1"], "lang": ["en"],
        "judge_pref": [0.3], "judge_pref_hard": [0.0],
        "position": ["A"], "opponent": ["anchor1"],
    })

    monkeypatch.setattr(sub, "sample_pool_battles", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sub, "judge_pool_battles", lambda *a, **k: new_battles_df)

    called = {}

    def _fake_place_against_pool(panel, new_model, new_battles, *, n_bootstraps, seed):
        called["place_against_pool"] = {"new_model": new_model}
        return ResultRecord(
            model=new_model, panel_version=panel.meta["panel_version"],
            panel_hash=panel.meta["panel_hash"], judge_model=panel.meta["judge_model"],
            elo_overall=1000.0, elo_std=0.0, elo_ci=[990.0, 1010.0], elo_per_lang={},
            winrate_overall=0.5, winrate_per_lang={}, n_battles=1,
            n_battles_per_lang={}, kappa_per_lang={}, mae_vs_human=float("nan"),
            scorer={}, generation_params={}, seed=seed,
        )

    monkeypatch.setattr(sub, "place_against_pool", _fake_place_against_pool)

    upload_calls = []

    def _fake_upload(**kwargs):
        upload_calls.append(kwargs)
        return "pr-url"

    monkeypatch.setattr(sub, "upload_folder", _fake_upload, raising=False)

    out = sub.main_submit([
        "--repo", "u/lb", "--model", "new_model",
        "--out", str(tmp_path / "results"),
        "--panel-version", "v1",
        "--dry-run",
        "--n-per-pair", "50",
    ])

    # place_against_pool must have been called
    assert "place_against_pool" in called
    assert called["place_against_pool"]["new_model"] == "new_model"

    # record must have been saved
    assert (out / "result.json").exists()

    # dry-run: no PR upload
    assert upload_calls == []


# ---------------------------------------------------------------------------
# Regression: _quantize_pref must be in PREFERENCE space (same direction as
# judge_pref), NOT win-for-model_a (which is the opposite convention used by
# pref_to_win_a).  A score < 0.5 means model_a is preferred → hard label 0.0.
# ---------------------------------------------------------------------------

def test_quantize_pref_preference_space_direction():
    import math

    from judgearena.leaderboard.curate import _quantize_pref

    assert _quantize_pref(0.1) == 0.0   # model_a preferred → 0.0, NOT 1.0
    assert _quantize_pref(0.9) == 1.0   # model_b preferred → 1.0, NOT 0.0
    assert _quantize_pref(0.5) == 0.5   # tie → 0.5
    assert math.isnan(_quantize_pref(float("nan")))  # NaN propagates
    assert math.isnan(_quantize_pref(None))          # None propagates as NaN
