"""Read-time assembly of the leaderboard bundle from records + cached anchors.

No Bradley-Terry, no inference stack: submission ELO is read from each record,
anchor ELO/calibration/h2h are read from the panel caches. This is what the
render Space imports.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

RNG_SEED_MAX = 2**31
CI_PERCENTILES = (2.5, 97.5)

_BOARD_COLUMNS = ["model", "elo", "ci_low", "ci_high", "n", "is_submission"]


class CalibrationPoint(TypedDict):
    model: str
    human_elo: float
    judge_elo: float
    judge_ci: list[float]


class CalibrationResult(TypedDict):
    mae: float
    spearman: float
    points: list[CalibrationPoint]


class AnchorRatings(TypedDict, total=False):
    overall: dict[str, float]
    per_lang: dict[str, dict[str, float]]
    counts_overall: dict[str, int]
    counts_per_lang: dict[str, dict[str, int]]
    winrate_overall: dict[str, float]


class Bundle(TypedDict):
    panel: dict
    languages: list[str]
    rows: list[dict]
    by_language: dict[str, list[dict]]
    calibration: CalibrationResult
    head_to_head: dict


def latest_panel_version(versions: list[str]) -> str:
    """Latest version, ordered by trailing integer then lexicographically (v10 > v9)."""
    if not versions:
        raise SystemExit("no panel versions found")

    def _key(v: str) -> tuple[int, str]:
        m = re.search(r"(\d+)$", v)
        return (int(m.group(1)) if m else -1, v)

    return sorted(versions, key=_key)[-1]


def record_label(rec: dict) -> str:
    return f"{rec['model']} #{rec['tag']}" if rec.get("tag") else rec["model"]


def _opt(value) -> float | None:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else float(value)


def pref_to_win_a(pref) -> float | None:
    """Soft pref → win for model_a: <0.5 win (1.0), >0.5 loss (0.0), ==0.5 tie (0.5), NaN/None → None."""
    if pref is None or (isinstance(pref, float) and np.isnan(pref)):
        return None
    return 1.0 if pref < 0.5 else (0.0 if pref > 0.5 else 0.5)


def board_rows(
    anchor_elo: dict[str, float],
    anchor_counts: dict[str, int],
    records: list[dict],
    panel_hash: str | None,
    *,
    lang: str | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for model, elo in anchor_elo.items():
        rows.append(
            {
                "model": model,
                "elo": float(elo),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n": int(anchor_counts.get(model, 0)),
                "is_submission": False,
            }
        )
    for rec in records:
        if rec.get("panel_hash") != panel_hash:
            continue
        if lang is None:
            elo = rec.get("elo_overall", float("nan"))
            ci = rec.get("elo_ci", [float("nan"), float("nan")])
            n = int(rec.get("n_battles", 0))
        else:
            elo = rec.get("elo_per_lang", {}).get(lang, float("nan"))
            ci = [float("nan"), float("nan")]
            n = int(rec.get("n_battles_per_lang", {}).get(lang, 0))
        rows.append(
            {
                "model": record_label(rec),
                "elo": float(elo),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
                "n": n,
                "is_submission": True,
            }
        )
    board = pd.DataFrame(rows, columns=_BOARD_COLUMNS)
    board = board.sort_values("elo", ascending=False, na_position="last").reset_index(drop=True)
    board.insert(0, "rank", range(1, len(board) + 1))
    return board


def _assemble_h2h(anchor_h2h: dict, records: list[dict], record_battles: dict, panel_hash: str | None) -> dict:
    wins: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row_model, cols in anchor_h2h.get("pairwise", {}).items():
        for col_model, (w, n) in cols.items():
            wins[(row_model, col_model)] += float(w)
            counts[(row_model, col_model)] += int(n)
    for rec in records:
        if rec.get("panel_hash") != panel_hash:
            continue
        battles = record_battles.get(record_label(rec))
        if battles is None or len(battles) == 0:
            continue
        if "opponent" not in battles.columns or "position" not in battles.columns:
            continue
        model = record_label(rec)
        for _, b in battles.iterrows():
            pref = b["judge_pref"]
            sub_win_a = pref_to_win_a(pref)
            if sub_win_a is None:
                continue
            if sub_win_a == 0.5:
                sub_win = 0.5
            elif (sub_win_a == 1.0 and b["position"] == "A") or (sub_win_a == 0.0 and b["position"] == "B"):
                sub_win = 1.0
            else:
                sub_win = 0.0
            opp = str(b["opponent"])
            wins[(model, opp)] += sub_win
            counts[(model, opp)] += 1
            wins[(opp, model)] += 1.0 - sub_win
            counts[(opp, model)] += 1
    models = sorted({m for pair in counts for m in pair})
    winrate = [
        [(wins[(row_model, col_model)] / counts[(row_model, col_model)]) if counts[(row_model, col_model)] else None for col_model in models]
        for row_model in models
    ]
    count_matrix = [[counts[(row_model, col_model)] for col_model in models] for row_model in models]
    return {"models": models, "winrate": winrate, "counts": count_matrix}


def assemble_bundle(
    panel_meta: dict,
    anchor_ratings: dict,
    calibration: dict,
    anchor_h2h: dict,
    records: list[dict],
    record_battles: dict,
) -> Bundle:
    panel_hash = panel_meta.get("panel_hash")
    overall = board_rows(
        anchor_ratings["overall"], anchor_ratings["counts_overall"], records, panel_hash
    )
    by_label = {record_label(r): r for r in records}
    anchor_ci = calibration.get("ci") or {
        p["model"]: p.get("judge_ci") for p in calibration.get("points", [])
    }
    anchor_winrate = anchor_ratings.get("winrate_overall", {})

    rows = []
    for _, row in overall.iterrows():
        entry = {
            "rank": int(row["rank"]),
            "model": row["model"],
            "elo": float(row["elo"]),
            "ci_low": _opt(row["ci_low"]),
            "ci_high": _opt(row["ci_high"]),
            "n": int(row["n"]),
            "is_submission": bool(row["is_submission"]),
            "winrate": None,
            "winrate_per_lang": {},
        }
        if row["is_submission"]:
            rec = by_label.get(row["model"])
            if rec is not None:
                entry["winrate"] = rec.get("winrate_overall")
                entry["winrate_per_lang"] = rec.get("winrate_per_lang", {})
        else:
            ci = anchor_ci.get(row["model"])
            if ci and len(ci) == 2:
                entry["ci_low"] = _opt(ci[0])
                entry["ci_high"] = _opt(ci[1])
            entry["winrate"] = anchor_winrate.get(row["model"])
        rows.append(entry)

    languages = sorted((panel_meta.get("kappa_per_language") or {}).keys())
    by_language: dict[str, list[dict]] = {}
    for lang in languages:
        lb = board_rows(
            anchor_ratings["per_lang"].get(lang, {}),
            anchor_ratings["counts_per_lang"].get(lang, {}),
            records,
            panel_hash,
            lang=lang,
        )
        by_language[lang] = [
            {
                "rank": int(r["rank"]),
                "model": r["model"],
                "elo": float(r["elo"]),
                "n": int(r["n"]),
                "is_submission": bool(r["is_submission"]),
            }
            for _, r in lb.iterrows()
        ]

    return {
        "panel": {
            "panel_version": panel_meta.get("panel_version"),
            "panel_hash": panel_hash,
            "judge_model": panel_meta.get("judge_model"),
            "baseline_model": panel_meta.get("baseline_model"),
            "mae_vs_human": panel_meta.get("mae_vs_human"),
            "kappa_per_language": panel_meta.get("kappa_per_language", {}),
            "scorer": panel_meta.get("scorer", {}),
            "generated_utc": datetime.now(UTC).isoformat(),
        },
        "languages": languages,
        "rows": rows,
        "by_language": by_language,
        "calibration": calibration,
        "head_to_head": _assemble_h2h(anchor_h2h, records, record_battles, panel_hash),
    }


def assemble_scores(records: list[dict], record_battles: dict, panel_hash: str | None) -> pd.DataFrame:
    cols = ["model", "tag", "lang", "judge_pref"]
    frames = []
    for rec in records:
        if rec.get("panel_hash") != panel_hash:
            continue
        battles = record_battles.get(record_label(rec))
        if battles is None or len(battles) == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "model": rec["model"],
                    "tag": rec.get("tag"),
                    "lang": battles["lang"].to_numpy(),
                    "judge_pref": battles["judge_pref"].to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


def load_records(records_root: str | Path) -> tuple[list[dict], dict]:
    records: list[dict] = []
    record_battles: dict = {}
    for result_path in sorted(Path(records_root).glob("*/result.json")):
        rec = json.loads(result_path.read_text())
        battles_path = result_path.parent / "battles.parquet"
        record_battles[record_label(rec)] = (
            pd.read_parquet(battles_path) if battles_path.exists() else None
        )
        records.append(rec)
    return records, record_battles


def load_anchor_caches(directory: str | Path) -> tuple[dict, dict, dict]:
    """Read the precomputed (anchor_ratings, calibration, anchor_h2h) caches.

    Pure JSON I/O — lives here (not in anchors.py) so the render Space never
    imports the curate-time Bradley-Terry stack just to read these files.
    """
    directory = Path(directory)
    ratings = json.loads((directory / "anchor_ratings.json").read_text())
    calibration = json.loads((directory / "calibration.json").read_text())
    h2h = json.loads((directory / "anchor_h2h.json").read_text())
    return ratings, calibration, h2h


def assemble_from_dirs(panel_dir: str | Path, records_root: str | Path) -> tuple[dict, pd.DataFrame]:
    """Build (bundle, scores) from a panel cache dir + a records/{version} dir."""
    panel_dir = Path(panel_dir)
    panel_meta = json.loads((panel_dir / "panel.json").read_text())
    anchor_ratings, calibration, anchor_h2h = load_anchor_caches(panel_dir)
    records, record_battles = load_records(records_root)
    bundle = assemble_bundle(
        panel_meta, anchor_ratings, calibration, anchor_h2h, records, record_battles
    )
    scores = assemble_scores(records, record_battles, panel_meta.get("panel_hash"))
    return bundle, scores
