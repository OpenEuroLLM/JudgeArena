import pandas as pd

from judgearena.tuning.budget import clip_trials_to_common_budget


def test_common_budget_counts_failed_panels_and_excludes_trials_after_horizon():
    def trials(rows):
        return pd.DataFrame(
            rows,
            columns=["neps_trial_id", "time_sampled", "battles_per_model", "status"],
        )

    runs = {
        "long": trials(
            [
                ("late", 1791835862.16, 5, "completed"),
                ("first", 1791835862.00, 5, "failed"),
                ("middle", 1791835862.08, 15, "completed"),
            ]
        ),
        "short": trials(
            [
                ("s2", 1791835862.08, 5, "failed"),
                ("s1", 1791835862.00, 5, "completed"),
            ]
        ),
    }
    clipped, info = clip_trials_to_common_budget(runs, {5: 57, 15: 164})

    assert info["logical_battle_horizon"] == 114
    assert clipped["long"].neps_trial_id.tolist() == ["first"]
    assert info["excluded_trial_ids_by_run"]["long"] == ["middle", "late"]
    assert clipped["short"].neps_trial_id.tolist() == ["s1", "s2"]
