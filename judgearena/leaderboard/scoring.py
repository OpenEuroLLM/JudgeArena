"""Score fixed comparisons with the run's ordinary metric requests."""

from judgearena.battles import hard_preferences, judged_battles, pairwise_aliases
from judgearena.benchmarks.scoring import calculate_metrics


def score_metrics(cfg, metrics, outcomes, model, *, human=False):
    if not metrics:
        return {}
    frame = outcomes.loc[
        (outcomes.model_a == model) | (outcomes.model_b == model)
    ].copy()
    if human:
        frame = frame.assign(
            evaluation_model=model,
            pref_hard=hard_preferences(frame.pref),
            **pairwise_aliases(frame, model_is_a=frame.model_a == model),
        )
    else:
        frame = judged_battles(
            frame,
            evaluation_model=model,
            judge=cfg.judge.model,
            judge_temperature=cfg.judge.temperature,
            judge_max_out_tokens=cfg.judge.max_out_tokens,
        )
    return calculate_metrics(
        frame,
        metrics,
        runtime={
            "seed": cfg.run.seed,
            "references": sorted(frame.baseline.unique().tolist()),
        },
    )
