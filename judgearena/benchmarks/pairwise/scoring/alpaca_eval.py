"""Official AlpacaEval 2.0 raw and length-controlled scoring."""

from __future__ import annotations

import math

import pandas as pd

from judgearena.benchmarks.pairwise.scoring.models import ScoringResult
from judgearena.log import get_logger
from judgearena.utils.eval import PrefSummary

logger = get_logger(__name__)

_INSTALL_HINT = (
    "The alpaca_eval_lc_winrate scorer requires the optional 'alpaca-eval' "
    "dependency; install it with `pip install 'judgearena[alpaca-eval]'`."
)


def check_requirements() -> None:
    """Fail before inference when the optional upstream scorer is unavailable."""
    try:
        import alpaca_eval.metrics  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _summarize(battles: pd.DataFrame) -> PrefSummary:
    """Return raw win rate over parsed battles."""
    prefs = battles["pref"]
    valid = prefs.dropna()
    return PrefSummary(
        num_battles=len(prefs),
        winrate=float((1 - valid).mean()) if len(valid) else float("nan"),
        num_wins=int((valid < 0.5).sum()),
        num_losses=int((valid > 0.5).sum()),
        num_ties=int((valid == 0.5).sum()),
        num_missing=int(prefs.isna().sum()),
    )


def _official_annotations(battles: pd.DataFrame) -> pd.DataFrame:
    """Map canonical battles onto AlpacaEval's annotation schema."""
    return pd.DataFrame(
        {
            "index": battles["instruction_index"].astype(int),
            "generator_1": battles["baseline"],
            "generator_2": battles["model"],
            "output_1": battles["completion_baseline"],
            "output_2": battles["completion_model"],
            "preference": (2 - battles["pref"]).fillna(0.0),
            "annotator": "judgearena",
        }
    )


def score(battles: pd.DataFrame) -> ScoringResult:
    """Return raw and official length-controlled AlpacaEval metrics."""
    from alpaca_eval.metrics import get_length_controlled_winrate

    if battles["instruction_index"].duplicated().any():
        logger.warning(
            "Battles contain duplicate instructions (swap_mode='both'?); the "
            "official LC protocol judges each instruction once in randomized "
            "order, so the LC winrate below is off-spec."
        )

    scoring_metrics: dict[str, float | None] = {
        "lc_winrate": None,
        "lc_standard_error": None,
        "raw_winrate": None,
    }
    try:
        upstream_metrics = get_length_controlled_winrate(
            _official_annotations(battles),
            save_weights_dir=None,
            is_add_glm_preference_inplace=False,
        )
    except Exception as exc:
        # Tiny subsets can break the GLM fit. Keep the raw summary and persist
        # the annotations so smoke runs remain useful.
        logger.warning("Length-controlled winrate computation failed: %s", exc)
        return ScoringResult(summary=_summarize(battles), metrics=scoring_metrics)

    for ours, theirs in (
        ("lc_winrate", "length_controlled_winrate"),
        ("lc_standard_error", "lc_standard_error"),
        ("raw_winrate", "win_rate"),
    ):
        value = upstream_metrics.get(theirs)
        if value is not None and math.isfinite(float(value)):
            scoring_metrics[ours] = float(value)
    if scoring_metrics["lc_winrate"] is None:
        logger.warning(
            "Length-controlled winrate is unavailable for this run (too few "
            "battles for a stable fit?)."
        )
    return ScoringResult(summary=_summarize(battles), metrics=scoring_metrics)
