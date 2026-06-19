"""Publish leaderboard results to a Hugging Face dataset bundle."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from judgearena.leaderboard.board import build_board
from judgearena.leaderboard.panel import Panel


def _record_label(rec: dict) -> str:
    return f"{rec['model']} #{rec['tag']}" if rec.get("tag") else rec["model"]


def _opt(value: float) -> float | None:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else float(value)


def build_bundle(panel: Panel, records: list[dict]) -> dict:
    """Assemble the published leaderboard bundle dict from a panel + records."""
    overall = build_board(panel, records)
    by_label = {_record_label(r): r for r in records}

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
        }
        if row["is_submission"]:
            rec = by_label.get(row["model"])
            if rec is not None:
                entry["winrate"] = rec.get("winrate_overall")
        rows.append(entry)

    languages = sorted((panel.meta.get("kappa_per_language") or {}).keys())
    by_language: dict[str, list[dict]] = {}
    for lang in languages:
        lb = build_board(panel, records, lang=lang)
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

    meta = panel.meta
    return {
        "panel": {
            "panel_version": meta.get("panel_version"),
            "panel_hash": meta.get("panel_hash"),
            "judge_model": meta.get("judge_model"),
            "baseline_model": meta.get("baseline_model"),
            "mae_vs_human": meta.get("mae_vs_human"),
            "kappa_per_language": meta.get("kappa_per_language", {}),
            "scorer": meta.get("scorer", {}),
            "generated_utc": datetime.now(UTC).isoformat(),
        },
        "languages": languages,
        "rows": rows,
        "by_language": by_language,
    }


def build_scores_frame(items: list[tuple[dict, pd.DataFrame]]) -> pd.DataFrame:
    """Long-form (model, tag, lang, judge_pref) from each (record, battles)."""
    cols = ["model", "tag", "lang", "judge_pref"]
    frames = []
    for rec, battles in items:
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
