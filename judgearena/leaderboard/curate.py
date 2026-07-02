"""Build and freeze a leaderboard panel from a human-labeled arena.

Pairs with ``leaderboard/score.py``, which scores a new model against the
frozen panel this module produces.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from judgearena.arenas_utils import _extract_instruction_text, load_arena_dataframe
from judgearena.config import EloArgs, JudgeArgs, PanelArgs
from judgearena.estimate_elo_ratings import _winner_to_pref
from judgearena.evaluate import (
    PairScore,
    calibrate_temperature,
    judge_and_parse_prefs,
    resolve_run_judge_prompt,
)
from judgearena.leaderboard.anchors import (
    human_elo_from_battles,
    judge_elo_from_battles,
    save_anchor_caches,
)
from judgearena.leaderboard.assemble import RNG_SEED_MAX
from judgearena.leaderboard.kappa import language_kappa
from judgearena.leaderboard.panel import PANEL_BATTLE_COLUMNS, Panel
from judgearena.leaderboard.pool import completions_from_battles, save_pool
from judgearena.log import get_logger
from judgearena.models import ChatVLLM, make_model

logger = get_logger(__name__)

MIN_CALIBRATION_PAIRS = 10


def select_roster(df: pd.DataFrame, args: PanelArgs) -> list[str]:
    """Resolve the anchor roster: explicit override, else coverage heuristic."""
    if args.roster_models is not None:
        return list(args.roster_models)

    long = pd.concat(
        [
            df[["model_a", "lang"]].rename(columns={"model_a": "model"}),
            df[["model_b", "lang"]].rename(columns={"model_b": "model"}),
        ],
        ignore_index=True,
    )
    grouped = long.groupby("model")
    counts = grouped.size()
    n_langs = grouped["lang"].nunique()

    qualified = [
        model
        for model in counts.index
        if counts[model] >= args.roster_min_annotations
        and n_langs[model] >= args.roster_min_languages
    ]
    qualified.sort(key=lambda m: -int(counts[m]))
    if args.roster_max_models is not None:
        qualified = qualified[: args.roster_max_models]
    return qualified


def _quantize_pref(pref: float | None) -> float:
    """Quantize a soft judge preference to {0.0, 0.5, 1.0} in PREFERENCE space
    (0.0 = model_a preferred), matching ``judge_pref``'s direction. This is the
    hardened preference used for ``judge_pref_hard`` and the κ gate — NOT
    ``pref_to_win_a`` (which is win-for-model_a, the opposite convention)."""
    if pref is None or (isinstance(pref, float) and np.isnan(pref)):
        return float("nan")
    if pref < 0.5:
        return 0.0
    if pref > 0.5:
        return 1.0
    return 0.5


def _battle_id(question_id: str, model_a: str, model_b: str) -> str:
    raw = f"{question_id}|{model_a}|{model_b}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _balanced_sample(
    group: pd.DataFrame, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample ``n`` battles keeping each model's battle count as equal as possible.

    Fills battles under a per-model cap (~2n/|models|), relaxing the cap when
    none more fit, so every model lands near the target and none is starved.
    """
    if len(group) <= n:
        return group.reset_index(drop=True)
    group = group.reset_index(drop=True)
    a = group["model_a"].to_numpy()
    b = group["model_b"].to_numpy()
    cap = max(1, round(2 * n / len(set(a) | set(b))))
    counts: dict[str, int] = {}
    order = rng.permutation(len(group))
    taken = np.zeros(len(group), dtype=bool)
    chosen: list[int] = []
    while len(chosen) < n:
        progressed = False
        for i in order:
            if len(chosen) >= n:
                break
            if taken[i] or counts.get(a[i], 0) >= cap or counts.get(b[i], 0) >= cap:
                continue
            counts[a[i]] = counts.get(a[i], 0) + 1
            counts[b[i]] = counts.get(b[i], 0) + 1
            taken[i] = True
            chosen.append(int(i))
            progressed = True
        if not progressed:
            cap += 1
    return group.iloc[chosen].reset_index(drop=True)


def build_panel(
    panel_args: PanelArgs,
    elo_args: EloArgs,
    *,
    judge_cfg: JudgeArgs,
    seed: int = 0,
    generation_params: dict | None = None,
) -> Panel:
    """Select the roster, judge κ-gated roster battles per language, and freeze the panel."""
    rng = np.random.default_rng(seed)
    df = load_arena_dataframe(arena=elo_args.arena)
    if elo_args.languages:
        df = df[df["lang"].isin(elo_args.languages)]

    roster = select_roster(df, panel_args)
    df = df[df["model_a"].isin(roster) & df["model_b"].isin(roster)].reset_index(drop=True)

    resolved = resolve_run_judge_prompt(elo_args.arena, judge_cfg)
    judge_model = make_model(model=judge_cfg.model)
    static_scorer = PairScore.from_resolved(
        resolved, temperature=elo_args.soft_elo_temperature
    )

    kept_frames: list[pd.DataFrame] = []
    kappa_per_language: dict[str, float] = {}
    n_per_language: dict[str, int] = {}

    for lang in sorted(df["lang"].unique()):
        group = df[df["lang"] == lang]
        if panel_args.balanced_per_model:
            group = _balanced_sample(group, panel_args.n_per_language, rng)
        elif len(group) > panel_args.n_per_language:
            group = group.sample(
                n=panel_args.n_per_language, random_state=int(rng.integers(0, RNG_SEED_MAX))
            )
        group = group.reset_index(drop=True)

        instructions = [_extract_instruction_text(c[0]) for c in group["conversation_a"]]
        completions_a = [_extract_instruction_text(c[1]) for c in group["conversation_a"]]
        completions_b = [_extract_instruction_text(c[1]) for c in group["conversation_b"]]

        annotations, _, prefs = judge_and_parse_prefs(
            judge_chat_model=judge_model,
            instructions=instructions,
            completions_A=completions_a,
            completions_B=completions_b,
            swap_mode=judge_cfg.swap_mode,
            provide_explanation=judge_cfg.provide_explanation,
            system_prompt=resolved.system_prompt,
            user_prompt_template=resolved.user_prompt_template,
            prompt_preset=resolved.preset_name,
            score_parser=static_scorer,
            use_tqdm=not isinstance(judge_model, ChatVLLM),
        )
        prefs = pd.Series(prefs).reset_index(drop=True)
        # Hard labels are temperature-independent; the κ gate runs before calibration.
        judge_hard = prefs.map(_quantize_pref)
        human_hard = group["winner"].map(_winner_to_pref)

        kappa = language_kappa(judge_hard, human_hard)
        if not (kappa > panel_args.kappa_threshold):
            logger.info("Dropping language %s (kappa=%.3f)", lang, kappa)
            continue
        kappa_per_language[lang] = kappa
        n_per_language[lang] = len(group)

        n = len(group)
        opp_is_a = rng.choice([True, False], size=n)
        chal_pos_a = rng.choice([True, False], size=n)
        frame = pd.DataFrame(
            {
                "battle_id": [
                    _battle_id(q, a, b)
                    for q, a, b in zip(group["question_id"], group["model_a"], group["model_b"], strict=True)
                ],
                "lang": lang,
                "model_a": group["model_a"].to_numpy(),
                "model_b": group["model_b"].to_numpy(),
                "instruction": instructions,
                "completion_a": completions_a,
                "completion_b": completions_b,
                "human_winner": group["winner"].to_numpy(),
                "challenger_opponent": [
                    a if opp_is_a[i] else b
                    for i, (a, b) in enumerate(zip(group["model_a"], group["model_b"], strict=True))
                ],
                "challenger_position": ["A" if chal_pos_a[i] else "B" for i in range(n)],
                # raw judge text kept transiently for calibration; dropped before freeze
                "judge_completion": [a.judge_completion for a in annotations],
            }
        )
        kept_frames.append(frame)

    battles = (
        pd.concat(kept_frames, ignore_index=True)
        if kept_frames
        else pd.DataFrame(columns=list(PANEL_BATTLE_COLUMNS))
    )

    # Global temperature calibration (once, across all kept battles) for
    # calibrated_soft; then re-derive the soft judge_pref at T and freeze.
    method = elo_args.method
    temperature = elo_args.soft_elo_temperature
    calibrated = False
    mae_vs_human = float("nan")
    if not battles.empty:
        if method == "calibrated_soft":
            delta_s: list[float] = []
            y: list[float] = []
            for jc, hw in zip(battles["judge_completion"], battles["human_winner"], strict=True):
                sa, sb = static_scorer.parse_raw_scores(jc)
                if sa is None or sb is None:
                    continue
                hp = _winner_to_pref(hw)
                if hp is None or hp == 0.5:
                    continue
                delta_s.append(sa - sb)
                y.append(1.0 - hp)  # pref=0 (A wins) -> y=1
            if len(delta_s) >= MIN_CALIBRATION_PAIRS:
                temperature = float(
                    calibrate_temperature(np.array(delta_s), np.array(y))
                )
                calibrated = True
            else:
                logger.info(
                    "Only %d calibration pairs (<%d); using static temperature.",
                    len(delta_s),
                    MIN_CALIBRATION_PAIRS,
                )

        soft_scorer = PairScore.from_resolved(resolved, temperature=temperature)
        judge_pref = battles["judge_completion"].map(soft_scorer.parse_model_raw)
        judge_pref = judge_pref.map(lambda x: float("nan") if x is None else float(x))
        battles["judge_pref"] = judge_pref.to_numpy()
        battles["judge_pref_hard"] = judge_pref.map(_quantize_pref).to_numpy()

        # MAE diagnostic: judge-ELO vs human-ELO over the anchor models.
        human_elo = human_elo_from_battles(battles, elo_args.baseline_model)
        judge_col = "judge_pref" if method != "hard" else "judge_pref_hard"
        judge_elo = judge_elo_from_battles(battles, judge_col, elo_args.baseline_model)
        overlap = [m for m in judge_elo if m in human_elo]
        if overlap:
            mae_vs_human = float(
                np.mean([abs(judge_elo[m] - human_elo[m]) for m in overlap])
            )

        # Persist per-axis criteria scores (JSON per row, A/B aligned to
        # model_a/model_b) before the raw judge_completion is dropped at freeze.
        # Single-score panels store null.
        if resolved.parser_mode == "criteria" and resolved.criteria_names:
            axis_names = list(resolved.criteria_names)
            battles["axis_scores"] = battles["judge_completion"].map(
                lambda jc: json.dumps(PairScore.parse_criteria_axes(jc, axis_names))
            )
        else:
            battles["axis_scores"] = None

        # Freeze exactly the schema columns (drops the transient judge_completion).
        battles = battles[list(PANEL_BATTLE_COLUMNS)].reset_index(drop=True)

    meta = {
        "panel_version": panel_args.panel_version,
        "arena": elo_args.arena,
        "judge_model": judge_cfg.model,
        "judge_prompt_preset": judge_cfg.prompt_preset,
        "criteria_file": judge_cfg.criteria_file,
        "swap_mode": judge_cfg.swap_mode,
        "baseline_model": elo_args.baseline_model,
        "roster": roster,
        "languages_kept": sorted(kappa_per_language),
        "n_per_language": n_per_language,
        "kappa_per_language": kappa_per_language,
        "kappa_threshold": panel_args.kappa_threshold,
        "seed": seed,
        "scorer": {"method": method, "temperature": temperature, "calibrated": calibrated},
        "mae_vs_human": mae_vs_human,
        "generation_params": generation_params or {},
    }
    return Panel(meta=meta, battles=battles)


def main_curate(argv: list[str] | None = None) -> None:
    """Curation entry: read a YAML (elo / panel / judge sections) and freeze a panel."""
    ap = argparse.ArgumentParser(prog="judgearena-curate-panel")
    ap.add_argument("--config_path", required=True)
    args = ap.parse_args(argv)
    with open(args.config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    elo_args = EloArgs(**data.get("elo", {}))
    panel_args = PanelArgs(**data.get("panel", {}))
    judge_cfg = JudgeArgs(**data["judge"])
    seed = int(data.get("run", {}).get("seed", 0))
    model = data.get("model", {})
    generation = data.get("generation", {})
    generation_params = {
        "max_out_tokens": model.get("max_out_tokens", 32768),
        "truncate_all_input_chars": generation.get("truncate_all_input_chars", 8192),
    }
    panel = build_panel(
        panel_args, elo_args, judge_cfg=judge_cfg, seed=seed,
        generation_params=generation_params,
    )
    models = sorted(set(panel.battles["model_a"]) | set(panel.battles["model_b"])) if not panel.battles.empty else []
    panel.meta["pool_models"] = models
    panel.meta["anchor_models"] = models
    completions = completions_from_battles(panel.battles)
    out = save_pool(panel, completions, f"{panel_args.panel_dir}/{panel_args.panel_version}")
    save_anchor_caches(panel, out)
    logger.info("Wrote panel + anchor caches to %s (%d battles)", out, len(panel.battles))

    repo = data.get("run", {}).get("dataset_repo")
    if repo:
        from huggingface_hub import upload_folder

        url = upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(out),
            path_in_repo=f"panel/{panel_args.panel_version}",
            commit_message=f"Add panel {panel_args.panel_version}",
        )
        logger.info("Uploaded panel to %s (%s)", repo, url)


if __name__ == "__main__":
    main_curate()
