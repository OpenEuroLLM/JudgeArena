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
    """Map parsed canonical battles onto AlpacaEval's annotation schema."""
    annotations = pd.DataFrame(
        {
            "index": battles["instruction_index"].astype(int),
            "generator_1": battles["baseline"],
            "generator_2": battles["model"],
            "output_1": battles["completion_baseline"],
            "output_2": battles["completion_model"],
            # AlpacaEval reserves 0 for its legacy draw encoding. Keep missing
            # parses as NaN rather than turning them into ties.
            "preference": 2 - battles["pref"],
            "annotator": "judgearena",
        }
    )
    # AlpacaEval 0.6.6 drops NaN labels from its raw metric and GLM training,
    # but still predicts the LC outcome for every input row. Remove missing
    # rows here so they cannot affect either metric.
    return annotations.dropna(subset=["preference"]).reset_index(drop=True)


def score(battles: pd.DataFrame) -> ScoringResult:
    """Return raw and official length-controlled AlpacaEval metrics."""
    from alpaca_eval.metrics import get_length_controlled_winrate

    if battles["instruction_index"].duplicated().any():
        logger.warning(
            "Battles contain duplicate instructions (swap_mode='both'?); the "
            "official LC protocol judges each instruction once in randomized "
            "order, so the LC winrate below is off-spec."
        )

    upstream_metrics = get_length_controlled_winrate(
        _official_annotations(battles),
        save_weights_dir=None,
        is_add_glm_preference_inplace=False,
    )

    scoring_metrics: dict[str, float] = {}
    for ours, theirs in (
        ("lc_winrate", "length_controlled_winrate"),
        ("lc_standard_error", "lc_standard_error"),
        ("raw_winrate", "win_rate"),
    ):
        value = upstream_metrics.get(theirs)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"Official AlpacaEval scoring did not return a finite {theirs!r}."
            )
        scoring_metrics[ours] = float(value)
    return ScoringResult(summary=_summarize(battles), metrics=scoring_metrics)
