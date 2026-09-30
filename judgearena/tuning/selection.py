"""Multi-objective survivor selection used by successive halving.

``pareto_efficiency_mask``, ``epsilon_net_indices`` and ``argsort_nondominated``
are adapted from judgetuning (github.com/geoalgo/judgetuning, Apache-2.0),
which took them from TSBench by Oliver Borchert.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as st


def pareto_efficiency_mask(X: np.ndarray) -> np.ndarray:
    """Mark rows of ``X`` not dominated by another row, all columns minimized."""
    mask = np.ones(X.shape[0], dtype=bool)
    for i, point in enumerate(X):
        if mask[i]:
            dominated = np.all(point <= X[mask], axis=1) & np.any(
                point < X[mask], axis=1
            )
            mask[mask] = ~dominated
    return mask


def epsilon_net_indices(X: np.ndarray, dim: int) -> np.ndarray:
    """Order rows so each next row is farthest from those already chosen.

    The seed is the row minimizing column ``dim`` (Clarkson, 2005, p. 17).
    """
    remaining = set(range(X.shape[0]))
    order = [int(np.argmin(X[:, dim]))]
    remaining.remove(order[0])
    while remaining:
        candidates = list(remaining)
        diff = X[candidates][:, None, :] - X[order][None, :, :]
        choice = candidates[np.linalg.norm(diff, axis=-1).min(-1).argmax()]
        order.append(choice)
        remaining.remove(choice)
    return np.array(order)


def argsort_nondominated(
    X: np.ndarray, *, minimize: list[bool], dim: int, max_items: int | None = None
) -> list[int]:
    """Sort rows by successive Pareto fronts, each ordered by an epsilon-net.

    Columns are quantile-normalized, so objectives on different scales weigh
    equally.
    """
    if X.shape[0] == 1:
        return [0]
    X = np.where(minimize, X, -X)
    X = (st.rankdata(X, axis=0) - 1) / (X.shape[0] - 1)
    remaining = np.arange(X.shape[0])
    indices: list[int] = []
    while remaining.size and (max_items is None or len(indices) < max_items):
        pareto_mask = pareto_efficiency_mask(X[remaining])
        front = remaining[pareto_mask]
        indices.extend(front[epsilon_net_indices(X[front], dim=dim)].tolist())
        remaining = remaining[~pareto_mask]
    return indices[:max_items]


def select_survivors(
    agreement: np.ndarray,
    cost: np.ndarray,
    *,
    n_keep: int,
    min_agreement: float | None = None,
) -> list[int]:
    """Return indices of the ``n_keep`` best trials by (agreement ↑, cost ↓).

    Trials at or below ``min_agreement`` are dropped first; the paper used a
    threshold just above a length-based baseline.
    """
    candidates = np.arange(len(agreement))
    if min_agreement is not None:
        candidates = candidates[agreement > min_agreement]
    if not candidates.size:
        return []
    order = argsort_nondominated(
        np.stack([cost[candidates], agreement[candidates]], axis=1),
        minimize=[True, False],
        dim=1,
        max_items=n_keep,
    )
    return candidates[order].tolist()
