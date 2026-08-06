import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from judgearena.arenas_utils import extract_turn_text, load_arena_dataframe
from judgearena.battles import Leaderboard, summarize_bootstrap, write_battles
from judgearena.benchmark import build_generation_kwargs
from judgearena.config import RunConfig, inference_cache_session
from judgearena.evaluate import (
    PairScore,
    calibrate_temperature,
    combine_swapped_prefs,
    judge_and_parse_prefs,
    resolve_run_judge_prompt,
)
from judgearena.generate import generate_instructions
from judgearena.inference_cache import InferenceCache
from judgearena.log import get_logger
from judgearena.models import build_default_judge_model_kwargs, make_model
from judgearena.repro import write_run_metadata
from judgearena.utils import compute_pref_summary
from judgearena.utils.eval import PrefSummary, Report

logger = get_logger(__name__)


def _winner_to_pref(winner: str) -> float | None:
    """Convert a hard winner label to a continuous preference value."""
    if winner == "model_a":
        return 0.0
    elif winner == "model_b":
        return 1.0
    elif winner in ("tie", "tie (bothbad)"):
        return 0.5
    return None


def _is_nan_pref(p) -> bool:
    return p is None or (isinstance(p, float) and np.isnan(p))


def fit_bradley_terry(
    df: pd.DataFrame,
    pref_col: str = "pref",
    scale: float = 400,
    base: float = 10,
    init_rating: float = 1000,
    baseline_model: str | None = None,
    baseline_rating: float = 1000,
) -> dict[str, float]:
    """Fit Bradley-Terry ratings via weighted logistic regression.

    Each row in *df* is a battle with columns ``model_a``, ``model_b`` and
    ``pref_col`` ∈ [0, 1] where 0 means A wins, 1 means B wins, 0.5 is a tie.
    Hard win/loss/tie labels are the special case ``pref ∈ {0, 0.5, 1}``.

    The soft cross-entropy for a battle is decomposed into two weighted
    hard-label rows so sklearn's ``LogisticRegression`` can be reused:

        Y=1, weight = (1 − pref) · count   (evidence A wins)
        Y=0, weight =  pref      · count   (evidence B wins)

    Identical ``(model_a, model_b, pref)`` triples are aggregated first so
    the design matrix stays small when prefs are quantised (e.g. human
    arena labels) and untouched when prefs are continuous floats.
    """
    df = df.dropna(subset=[pref_col])
    if df.empty:
        return {}

    grouped = (
        df.groupby(["model_a", "model_b", pref_col]).size().reset_index(name="count")
    )

    all_models = sorted(set(grouped["model_a"]) | set(grouped["model_b"]))
    models = pd.Series(np.arange(len(all_models)), index=all_models)
    p = len(models)

    m_a_idx = grouped["model_a"].map(models).to_numpy()
    m_b_idx = grouped["model_b"].map(models).to_numpy()
    prefs = grouped[pref_col].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n = len(grouped)

    log_base = np.log(base)
    X = np.zeros((2 * n, p))
    top = np.arange(n)
    bot = n + top
    X[top, m_a_idx] = +log_base
    X[top, m_b_idx] = -log_base
    X[bot, m_a_idx] = +log_base
    X[bot, m_b_idx] = -log_base

    Y = np.concatenate([np.ones(n), np.zeros(n)])
    sample_weights = np.concatenate([(1.0 - prefs) * counts, prefs * counts])

    # Keep zero-weight rows so sklearn LR always sees both Y classes — when
    # every pref collapses to 0 or 1 the missing-class rows contribute nothing
    # to the loss but stop the solver from raising on n_classes < 2.
    if sample_weights.sum() == 0:
        return {}

    lr = LogisticRegression(fit_intercept=False, C=1e10, tol=1e-6, max_iter=1000)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = scale * lr.coef_[0] + init_rating

    if baseline_model is not None and baseline_model in models.index:
        elo_scores += baseline_rating - elo_scores[models[baseline_model]]

    return dict(pd.Series(elo_scores, index=models.index))


def _sample_fingerprint(sampled: pd.DataFrame) -> str:
    rows = []
    for index, row in sampled.iterrows():
        rows.append(
            {
                "index": int(index)
                if isinstance(index, int | np.integer)
                else str(index),
                "question_id": str(row["question_id"]),
                "model_a": str(row["model_a"]),
                "model_b": str(row["model_b"]),
            }
        )
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def select_seeded_random_arena_battles(
    df_battles: pd.DataFrame,
    *,
    n_battles: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select a shared random battle panel for outside-model Elo estimation."""
    n = min(n_battles, len(df_battles))
    sampled = df_battles.sample(n=n, random_state=seed, replace=False)
    metadata: dict[str, object] = {
        "sampling_mode": "seeded_random",
        "random_seed": seed,
        "requested_rows": n_battles,
        "sampled_rows": len(sampled),
        "sampled_original_indices": [
            int(index) if isinstance(index, int | np.integer) else str(index)
            for index in sampled.index
        ],
        "sampled_question_ids": [
            str(value) for value in sampled["question_id"].tolist()
        ],
        "sample_fingerprint": _sample_fingerprint(sampled),
    }
    return sampled.reset_index(drop=True), metadata


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "model"


class EloReport(Report):
    """Bradley-Terry / Soft-ELO ratings for one focal model against an arena.

    This is the console/``results-*.json`` run report. The narrower per-model
    leaderboard with bootstrap CIs is persisted separately as
    ``elo_ratings.json`` via :class:`judgearena.battles.Leaderboard`.
    """

    arena: str
    """Arena/benchmark the focal model is rated against."""
    judge_model: str
    """LLM judge that scored the battles."""
    summary: PrefSummary
    """Win/loss/tie stats for the focal model's LLM-judged battles."""
    num_battles: int
    """Total battles (LLM-judged + human-anchor)."""
    llm_judged_battles: int
    """Battles the LLM judged for the focal model."""
    human_anchor_battles: int
    """Human-annotated battles anchoring the other arena models."""
    elo_mean: float
    """Focal model's mean ELO across bootstrap samples."""
    elo_std: float
    """Std of the focal model's ELO across bootstrap samples."""
    elo_n_bootstraps: int
    """Bootstrap samples that rated the focal model (≤ n_bootstraps); the n behind elo_mean/elo_std."""
    mae_vs_human: float
    """Mean absolute error of estimated vs human ELO over overlapping models."""
    method: str
    """Rating method label (e.g. "Soft-ELO")."""
    n_bootstraps: int
    """Total bootstrap iterations run."""
    model_name: str
    """Focal model under evaluation."""
    mean_ratings: dict[str, float]
    """Per-model mean ELO across bootstraps."""
    battle_counts: dict[str, int]
    """Per-model battle count."""
    human_elo: dict[str, float]
    """Per-model human-derived ELO (anchors)."""
    bootstrap_ratings: list[dict[str, float]]
    """One model→ELO dict per bootstrap sample."""
    sampling_metadata: dict[str, object]
    """Instruction-sampling parameters for the run."""

    def render(self) -> None:
        s = self.summary
        print(f"\n=== Results for {self.model_name} ===")
        print(
            f"Battles: {self.llm_judged_battles} | Wins: {s.num_wins} | "
            f"Losses: {s.num_losses} | Ties: {s.num_ties}"
        )
        print(f"Win rate: {s.winrate:.2%}")

        print(
            f"\n=== {self.method} Ratings (Bradley-Terry, "
            f"{self.n_bootstraps} bootstraps) ==="
        )
        print(
            f"Estimating {self.method} Ratings with {self.llm_judged_battles} "
            f"LLM-judges for model {self.model_name} and {self.human_anchor_battles} "
            "human annotations for other models. Number of battles is indicated in "
            "parenthesis and confidence intervals are reported by computing ELO on "
            f"{self.n_bootstraps} samples of instructions."
        )

        if not self.mean_ratings:
            print("  Not enough data to compute ELO ratings.")
            return

        # Percentile CIs (not mean ± std): matches the bounds persisted in
        # elo_ratings.json so the console and the saved leaderboard never disagree.
        for e in summarize_bootstrap(
            self.bootstrap_ratings, self.battle_counts, self.model_name
        ):
            suffix = " <-----" if e.model == self.model_name else ""
            print(
                f"  {e.model}  ({e.n_battles}){suffix}: "
                f"{e.rating:.1f} [{e.ci_low:.1f}, {e.ci_high:.1f}]"
            )

        overlap = [
            m for m in self.mean_ratings if m in self.human_elo and m != self.model_name
        ]
        if overlap:
            print(
                f"\n  MAE vs Human-ELO ({len(overlap)} arena models): "
                f"{self.mae_vs_human:.1f}"
            )
        else:
            print("\n  No overlapping arena models to compute MAE.")


def _prefs_to_battle_results(
    prefs,
    our_model_is_position_a,
    opponent_models,
    model_name: str,
    *,
    judge_model: str | None = None,
    question_ids=None,
) -> pd.DataFrame:
    """Map per-battle judge prefs into model-name-level battle rows.

    The judge prompt placed our model at position A or B independently per
    battle.  Here we re-orient each row so ``model_a``/``model_b`` carry
    the actual model names and ``pref`` is consistent with that ordering
    (``pref=0`` ⇒ ``model_a`` wins).  ``pref_hard`` is the quantised
    {0, 0.5, 1} version used by the non-soft Bradley-Terry fit.
    """
    records = []
    for pref, is_pos_a, opp in zip(
        prefs, our_model_is_position_a, opponent_models, strict=True
    ):
        if _is_nan_pref(pref) or pref == 0.5:
            winner = "tie"
        elif pref < 0.5:
            winner = "model_a"
        else:
            winner = "model_b"

        if is_pos_a:
            rec = {
                "model_a": model_name,
                "model_b": opp,
                "winner": winner,
                "pref": pref,
            }
        else:
            rec = {
                "model_a": opp,
                "model_b": model_name,
                "winner": winner,
                "pref": None if _is_nan_pref(pref) else pref,
            }
        rec["pref_hard"] = _winner_to_pref(winner)
        records.append(rec)
    df = pd.DataFrame(records)
    df["source"] = "llm-judge"
    df["judge_model"] = judge_model
    if question_ids is not None:
        df["question_id"] = question_ids
    return df


def _battle_identity_fallback(
    *,
    instruction: str,
    focal_model: str,
    opponent_model: str,
    position: str,
) -> str:
    payload = {
        "instruction": instruction,
        "focal_model": focal_model,
        "opponent_model": opponent_model,
        "position": position,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _elo_generation_row_metadata(
    *,
    arena: str,
    df_battles: pd.DataFrame,
    instructions: list[str],
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for index, instruction in enumerate(instructions):
        row: dict[str, Any] = {
            "arena": arena,
            "source": "elo-generation",
        }
        question_id = (
            df_battles.iloc[index]["question_id"]
            if "question_id" in df_battles.columns
            else None
        )
        if question_id is not None and pd.notna(question_id):
            row["question_id"] = str(question_id)
        else:
            row["instruction_sha256"] = hashlib.sha256(
                instruction.encode("utf-8")
            ).hexdigest()
        metadata.append(row)
    return metadata


def _elo_judge_row_metadata(
    *,
    arena: str,
    df_battles: pd.DataFrame,
    instructions: list[str],
    focal_model: str,
    opponent_models: list[str],
    our_model_is_position_a: np.ndarray,
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for index, opponent_model in enumerate(opponent_models):
        position = "A" if our_model_is_position_a[index] else "B"
        row: dict[str, Any] = {
            "arena": arena,
            "source": "elo-judge",
            "focal_model": focal_model,
            "opponent_model": opponent_model,
            "position": position,
        }
        question_id = (
            df_battles.iloc[index]["question_id"]
            if "question_id" in df_battles.columns
            else None
        )
        if question_id is not None and pd.notna(question_id):
            row["question_id"] = str(question_id)
        else:
            row["battle_identity"] = _battle_identity_fallback(
                instruction=instructions[index],
                focal_model=focal_model,
                opponent_model=opponent_model,
                position=position,
            )
        metadata.append(row)
    return metadata


def _elo_calibration_row_metadata(
    *,
    arena: str,
    cal_battles: pd.DataFrame,
) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for row_index in cal_battles.index:
        row: dict[str, Any] = {
            "arena": arena,
            "source": "elo-calibration",
            "purpose": "temperature_calibration",
        }
        question_id = (
            cal_battles.loc[row_index, "question_id"]
            if "question_id" in cal_battles.columns
            else None
        )
        if question_id is not None and pd.notna(question_id):
            row["question_id"] = str(question_id)
        else:
            row["arena_row_index"] = (
                int(row_index)
                if isinstance(row_index, int | np.integer)
                else str(row_index)
            )
        metadata.append(row)
    return metadata


def _parse_prefs_from_judge_completions(
    judge_completions: list[str],
    *,
    swap_mode: str,
    score_parser: PairScore,
) -> list[float]:
    parsed = pd.Series(
        [score_parser.parse_model_raw(completion) for completion in judge_completions]
    ).apply(lambda value: float("nan") if value is None else value)
    if swap_mode == "both":
        n_half = len(judge_completions) // 2
        return combine_swapped_prefs(parsed[:n_half], parsed[n_half:]).tolist()
    return parsed.tolist()


def arena_anchor_battles(df_arena_all: pd.DataFrame) -> pd.DataFrame:
    """Human anchor battles from a loaded arena frame.

    Keeps battles between models with at least 500 human votes and shapes them
    like the persisted rows (``pref``/``pref_hard`` from the hard winner label,
    ``source="human"``).

    These anchors are a deterministic function of the (revision-pinned) arena,
    so runs persist only their own llm-judge battles; recompute ELO with
    ``arena_anchor_battles(load_arena_dataframe(arena))`` + the run's saved
    ``battles.parquet``. The original ``df_arena_all`` index is preserved so
    callers can still look up the full conversation rows by row label.
    """
    df = df_arena_all.loc[:, ["model_a", "model_b", "winner"]].copy()
    counts = pd.concat([df["model_a"], df["model_b"]]).value_counts()
    well_represented = set(counts[counts >= 500].index)
    df = df[df["model_a"].isin(well_represented) & df["model_b"].isin(well_represented)]
    # Hard human labels → pref ∈ {0, 0.5, 1}; pref_hard == pref.
    df["pref"] = df["winner"].map(_winner_to_pref)
    df["pref_hard"] = df["pref"]
    df["source"] = "human"
    return df


def main(cfg: RunConfig) -> dict:
    assert cfg.elo is not None  # main is dispatched only for elo tasks
    with inference_cache_session(cfg) as cache:
        return _run_elo(cfg, cache=cache)


def _run_elo(cfg: RunConfig, *, cache: InferenceCache | None) -> dict:
    assert cfg.elo is not None
    run_started_at = datetime.now(UTC)
    rng = np.random.default_rng(cfg.run.seed)

    # Step 1: Load arena battles
    logger.info("Step 1: Loading battles from %s", cfg.elo.arena)
    df_arena_all = load_arena_dataframe(arena=cfg.elo.arena)

    # Filter by language if specified
    df_battles = df_arena_all
    if cfg.elo.languages:
        df_battles = df_battles[df_battles["lang"].isin(cfg.elo.languages)]

    random_sampling = cfg.elo.elo_random_battles is not None
    sampling_metadata: dict[str, object] = {"sampling_mode": "head"}
    if random_sampling:
        if (
            cfg.generation.n_instructions is not None
            or cfg.elo.n_instructions_per_language is not None
        ):
            raise ValueError(
                "n_instructions and n_instructions_per_language cannot be combined "
                "with elo_random_battles."
            )
        df_battles, sampling_metadata = select_seeded_random_arena_battles(
            df_battles,
            n_battles=cfg.elo.elo_random_battles,
            seed=cfg.run.seed,
        )
    else:
        # Keep at most n_instructions_per_language per language
        if cfg.elo.n_instructions_per_language is not None:
            df_battles = (
                df_battles.groupby("lang")
                .head(cfg.elo.n_instructions_per_language)
                .reset_index(drop=True)
            )

        # Keep at most n_instructions total (subset used for LLM-judge evaluation)
        if cfg.generation.n_instructions is not None:
            df_battles = df_battles.head(cfg.generation.n_instructions)

    df_battles = df_battles.reset_index(drop=True)
    n = len(df_battles)
    logger.info("Loaded %d battles.", n)

    # Extract user instructions (first turn of conversation_a)
    instructions = pd.Series(
        [
            extract_turn_text(row["conversation_a"][0])
            for _, row in df_battles.iterrows()
        ],
        name="instruction",
    )
    logger.debug("First instruction:\n%s", instructions.iloc[0][:300])

    # Step 2: Generate completions for the model under evaluation
    logger.info("Step 2: Generating completions with %s", cfg.model.name)

    # Mirror the benchmark generation path so Elo battles honor the
    # thinking-token sub-budget for thinking models (the Elo entrypoint
    # previously called evaluated_generation_kwargs() directly and silently
    # dropped battle_thinking_token_budget).
    extra_kwargs = build_generation_kwargs(cfg, cfg.model.name, role="A")
    use_tqdm = cfg.run.use_tqdm
    instruction_text = instructions.tolist()
    completions_df = generate_instructions(
        instructions=instructions,
        model=cfg.model.name,
        truncate_input_chars=cfg.generation.truncate_all_input_chars,
        use_tqdm=use_tqdm,
        cache=cache,
        row_metadata=_elo_generation_row_metadata(
            arena=cfg.elo.arena,
            df_battles=df_battles,
            instructions=instruction_text,
        ),
        **extra_kwargs,
    ).set_index("instruction_index")
    completions = completions_df.loc[:, "completion"]

    logger.debug("First completion:\n%s", completions.iloc[0])

    # Step 3: Judge evaluation against randomly picked arena opponents
    logger.info("Step 3: Judge evaluation with %s", cfg.judge.model)

    # For each battle, randomly pick opponent: model_a or model_b from the arena
    use_model_a_as_opponent = rng.choice([True, False], size=n)
    # Randomly decide if our model is in position A or B for the judge
    our_model_is_position_a = rng.choice([True, False], size=n)

    opponent_completions = [
        (
            extract_turn_text(row["conversation_a"][1])
            if use_model_a_as_opponent[i]
            else extract_turn_text(row["conversation_b"][1])
        )
        for i, (_, row) in enumerate(df_battles.iterrows())
    ]
    opponent_models = [
        row["model_a"] if use_model_a_as_opponent[i] else row["model_b"]
        for i, (_, row) in enumerate(df_battles.iterrows())
    ]

    our_completions = completions.tolist()
    resolved_prompt = resolve_run_judge_prompt(cfg.elo.arena, cfg.judge)

    completions_A = [
        our_completions[i] if our_model_is_position_a[i] else opponent_completions[i]
        for i in range(n)
    ]
    completions_B = [
        opponent_completions[i] if our_model_is_position_a[i] else our_completions[i]
        for i in range(n)
    ]

    judge_extra_kwargs = build_default_judge_model_kwargs(
        cfg.judge.model,
        cfg.model.engine_kwargs,
        judge_engine_kwargs_override=cfg.judge.model_kwargs(
            fallback_chat_template=cfg.model.chat_template,
        ),
    )
    judge_chat_model = make_model(
        model=cfg.judge.model,
        **judge_extra_kwargs,
    )
    row_metadata = _elo_judge_row_metadata(
        arena=cfg.elo.arena,
        df_battles=df_battles,
        instructions=instruction_text,
        focal_model=cfg.model.name,
        opponent_models=opponent_models,
        our_model_is_position_a=our_model_is_position_a,
    )
    annotations, annotations_reversed, _ = judge_and_parse_prefs(
        judge_chat_model=judge_chat_model,
        instructions=instruction_text,
        completions_A=completions_A,
        completions_B=completions_B,
        swap_mode=cfg.judge.swap_mode,
        provide_explanation=cfg.judge.provide_explanation,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        system_prompt=resolved_prompt.system_prompt,
        user_prompt_template=resolved_prompt.user_prompt_template,
        prompt_preset=resolved_prompt.preset_name,
        parser_mode=resolved_prompt.parser_mode,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        use_tqdm=use_tqdm,
        cache=cache,
        row_metadata=row_metadata,
    )
    if annotations_reversed is None:
        row_annotations = list(annotations)
    else:
        row_annotations = list(annotations) + list(annotations_reversed)
    judge_completions = [annotation.judge_completion for annotation in row_annotations]

    if "question_id" in df_battles.columns and len(df_battles):
        qids = df_battles["question_id"].tolist()
        n_rep = (len(row_annotations) + len(qids) - 1) // len(qids)
        question_ids = (qids * n_rep)[: len(row_annotations)]
    else:
        question_ids = [None] * len(row_annotations)

    logger.debug("First judge output:\n%s", judge_completions[0][:500])

    model_name = cfg.model.name

    # Anchor the llm-judge battles against the human arena battles. These are
    # rebuilt from the (revision-pinned) arena, not persisted per run.
    df_arena = arena_anchor_battles(df_arena_all)

    # Compute human-only BT ratings as ground-truth reference
    human_elo = fit_bradley_terry(
        df_arena, pref_col="pref_hard", baseline_model=cfg.elo.baseline_model
    )

    # --- Temperature calibration (optional) ---
    # Run the judge on a random subset of human arena battles that already
    # have ground-truth winner labels so we can fit T* via MLE.
    calibrated_temperature: float | None = None
    if cfg.elo.calibrate_temperature:
        if not cfg.elo.soft_elo:
            logger.warning(
                "--calibrate-temperature has no effect with --no-soft-elo; skipping."
            )
        else:
            logger.info("Calibrating PairScore temperature against human annotations.")
            # Sample calibration battles from the already-loaded arena battles.
            # Use the same judge to score them so scores and labels are comparable.
            _cal_n = (
                min(cfg.elo.calibration_size, len(df_arena))
                if cfg.elo.calibration_size is not None
                else len(df_arena)
            )
            # Keep the original df_arena_all index so we can look up the full
            # conversation rows below; reset_index would point at non-existent
            # 0..N labels in df_arena_all.
            cal_battles = df_arena.sample(
                n=_cal_n, random_state=int(rng.integers(0, 2**31))
            )

            cal_instructions = [
                extract_turn_text(df_arena_all.loc[i, "conversation_a"][0])
                for i in cal_battles.index
            ]
            cal_completions_a = [
                extract_turn_text(df_arena_all.loc[i, "conversation_a"][1])
                for i in cal_battles.index
            ]
            cal_completions_b = [
                extract_turn_text(df_arena_all.loc[i, "conversation_b"][1])
                for i in cal_battles.index
            ]

            cal_row_metadata = _elo_calibration_row_metadata(
                arena=cfg.elo.arena,
                cal_battles=cal_battles,
            )
            cal_annotations, _, _ = judge_and_parse_prefs(
                judge_chat_model=judge_chat_model,
                instructions=cal_instructions,
                completions_A=cal_completions_a,
                completions_B=cal_completions_b,
                swap_mode=cfg.judge.swap_mode,
                provide_explanation=cfg.judge.provide_explanation,
                strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
                system_prompt=resolved_prompt.system_prompt,
                user_prompt_template=resolved_prompt.user_prompt_template,
                prompt_preset=resolved_prompt.preset_name,
                parser_mode=resolved_prompt.parser_mode,
                truncate_input_chars=cfg.generation.truncate_judge_input_chars,
                use_tqdm=use_tqdm,
                cache=cache,
                row_metadata=cal_row_metadata,
            )

            # Build (delta_s, y) pairs from calibration battles.
            # delta_s = score_A - score_B, extracted exactly as the main run does.
            delta_s_cal = []
            y_cal = []
            for ann, human_winner in zip(
                cal_annotations, cal_battles["winner"].tolist(), strict=True
            ):
                sa, sb = PairScore.parse_raw_scores(ann.judge_completion)
                if sa is None or sb is None:
                    continue
                human_pref = _winner_to_pref(human_winner)
                if human_pref is None or human_pref == 0.5:
                    continue  # skip ties and missing
                delta_s_cal.append(sa - sb)
                y_cal.append(1.0 - human_pref)  # pref=0 → A wins → y=1

            if len(delta_s_cal) < 10:
                logger.warning(
                    "Only %d valid calibration pairs (need ≥10); keeping default temperature.",
                    len(delta_s_cal),
                )
            else:
                calibrated_temperature = calibrate_temperature(
                    np.array(delta_s_cal), np.array(y_cal)
                )
                logger.info(
                    "Calibration pairs: %d  T* = %.4f  (default was %s)",
                    len(delta_s_cal),
                    calibrated_temperature,
                    cfg.elo.soft_elo_temperature,
                )

    score_parser = PairScore(
        temperature=calibrated_temperature
        if calibrated_temperature is not None
        else cfg.elo.soft_elo_temperature,
        parser_mode=resolved_prompt.parser_mode,
    )
    prefs = _parse_prefs_from_judge_completions(
        judge_completions,
        swap_mode=cfg.judge.swap_mode,
        score_parser=score_parser,
    )

    if cfg.judge.swap_mode == "both":
        battle_our_pos_a = np.concatenate(
            [our_model_is_position_a, our_model_is_position_a]
        )
        battle_opponents = list(opponent_models) + list(opponent_models)
    else:
        battle_our_pos_a = our_model_is_position_a
        battle_opponents = opponent_models

    df_llm_judge = _prefs_to_battle_results(
        prefs,
        battle_our_pos_a,
        battle_opponents,
        model_name,
        judge_model=cfg.judge.model,
        question_ids=question_ids,
    )
    df_results = pd.concat([df_llm_judge, df_arena], ignore_index=True)

    prefs_normalized = pd.Series(
        [
            p if (p is None or is_pos_a) else (1 - p)
            for p, is_pos_a in zip(prefs, battle_our_pos_a, strict=True)
        ]
    )
    summary = compute_pref_summary(prefs_normalized)

    n_bootstraps = cfg.elo.n_bootstraps
    use_soft = cfg.elo.soft_elo

    n_llm = len(df_llm_judge)
    n_human = len(df_arena)
    method_label = "Soft-ELO" if use_soft else "ELO"

    # Count battles per model across the combined results
    battle_counts: dict[str, int] = {}
    for _, row in df_results.iterrows():
        battle_counts[row["model_a"]] = battle_counts.get(row["model_a"], 0) + 1
        battle_counts[row["model_b"]] = battle_counts.get(row["model_b"], 0) + 1

    pref_col = "pref" if use_soft else "pref_hard"
    bootstrap_ratings: list[dict[str, float]] = []
    for _ in range(n_bootstraps):
        df_sample = df_results.sample(
            n=len(df_results), replace=True, random_state=int(rng.integers(0, 2**31))
        )
        ratings = fit_bradley_terry(
            df_sample, pref_col=pref_col, baseline_model=cfg.elo.baseline_model
        )
        bootstrap_ratings.append(ratings)

    # One percentile-CI summary, reused for the console report, the MAE
    # calculation below, and the persisted elo_ratings.json leaderboard so
    # none of the three can disagree.
    entries: list = []
    mean_ratings: dict[str, float] = {}
    mae = np.nan
    if bootstrap_ratings:
        entries = summarize_bootstrap(bootstrap_ratings, battle_counts, model_name)
        mean_ratings = {e.model: e.rating for e in entries}
        overlap = [m for m in mean_ratings if m in human_elo and m != model_name]
        if overlap:
            abs_errors = [abs(mean_ratings[m] - human_elo[m]) for m in overlap]
            mae = np.mean(abs_errors)

    model_rating_values = [
        rating[model_name] for rating in bootstrap_ratings if model_name in rating
    ]
    elo_mean = (
        float(np.mean(model_rating_values)) if model_rating_values else float("nan")
    )
    elo_std = (
        float(np.std(model_rating_values)) if model_rating_values else float("nan")
    )

    report = EloReport(
        arena=cfg.elo.arena,
        judge_model=cfg.judge.model,
        summary=summary,
        num_battles=n,
        llm_judged_battles=n_llm,
        human_anchor_battles=n_human,
        elo_mean=elo_mean,
        elo_std=elo_std,
        elo_n_bootstraps=len(model_rating_values),
        mae_vs_human=mae,
        method=method_label,
        n_bootstraps=n_bootstraps,
        model_name=model_name,
        mean_ratings=mean_ratings,
        battle_counts=battle_counts,
        human_elo=human_elo,
        bootstrap_ratings=bootstrap_ratings,
        sampling_metadata=sampling_metadata,
    )
    results = report.to_dict()
    report.render()
    # ELO artifacts (ratings, battles, bootstrap CSV, metadata) are judge-specific,
    # so key the folder on the judge too — otherwise re-running the same
    # arena/model under a different judge silently overwrites the previous run.
    result_path = report.save(
        Path(cfg.run.result_folder)
        / f"elo-{_slugify(cfg.elo.arena)}-{_slugify(model_name)}-{_slugify(cfg.judge.model)}"
        / f"results-{_slugify(model_name)}.json"
    )

    # Persist only the run's own llm-judge battles (a few KB). The human arena
    # anchors are identical across every run, so we do not duplicate them per
    # experiment — recompute ELO by recombining with
    # arena_anchor_battles(load_arena_dataframe(arena)). question_id is the
    # instruction-index join key back to the arena initial table / completion
    # cache. battles.parquet keeps pref_hard so both hard and soft ELO recompute.
    res_dir = result_path.parent
    battle_cols = [
        "model_a",
        "model_b",
        "winner",
        "pref",
        "pref_hard",
        "source",
        "judge_model",
        "question_id",
    ]
    write_battles(
        res_dir / "battles.parquet",
        df_llm_judge[[c for c in battle_cols if c in df_llm_judge.columns]],
    )
    if bootstrap_ratings:
        pd.DataFrame(bootstrap_ratings).to_csv(
            res_dir / "bootstrap_ratings.csv", index=False
        )
        Leaderboard(
            arena=cfg.elo.arena,
            model=model_name,
            judge_model=cfg.judge.model,
            n_bootstraps=n_bootstraps,
            seed=cfg.run.seed,
            ratings=entries,
        ).write(res_dir / "elo_ratings.json")

    # Reproducibility manifest (git hash, dependency versions, timings) — parity
    # with the other entrypoints, all of which write run-metadata. Best-effort:
    # a metadata-write failure should not sink an already-completed run.
    try:
        write_run_metadata(
            output_dir=res_dir,
            entrypoint="judgearena.estimate_elo_ratings.main",
            run=cfg.model_dump(),
            results=results,
            input_payloads=(
                {"question_id": df_battles["question_id"].tolist()}
                if "question_id" in df_battles.columns
                else None
            ),
            judge_system_prompt=resolved_prompt.system_prompt,
            judge_user_prompt_template=resolved_prompt.user_prompt_template,
            started_at_utc=run_started_at,
        )
    except OSError as e:
        logger.warning("Failed to write run metadata: %s", e)

    return {
        **results,
        "result_path": str(result_path),
    }
