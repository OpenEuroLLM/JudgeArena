"""Flow A: curate and freeze the leaderboard panel."""

from __future__ import annotations

import pandas as pd

from judgearena.config import PanelArgs


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
