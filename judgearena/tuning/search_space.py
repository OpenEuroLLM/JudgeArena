"""Expand a judge-tuning search space into concrete trial overrides."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

TUNER_OWNED_KEYS = frozenset(
    {"task", "meta_eval.battles_per_model", "meta_eval.split", "run.result_folder"}
)


@dataclass(frozen=True)
class Trial:
    """One judge configuration, given as dotted ``RunConfig`` overrides."""

    overrides: dict[str, object]

    @property
    def id(self) -> str:
        payload = json.dumps(self.overrides, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def expand_grid(search_space: Mapping[str, Sequence[object]]) -> list[Trial]:
    """Return the cartesian product of the search-space axes.

    An axis name is the dotted ``RunConfig`` path its values override. An axis
    whose values are mappings is grouped instead: each value is a set of dotted
    overrides that change together, and the axis name is only a label.
    """
    axes = [
        [
            dict(value) if isinstance(value, Mapping) else {name: value}
            for value in values
        ]
        for name, values in search_space.items()
    ]
    keys = {key for choices in axes for choice in choices for key in choice}
    owned = sorted(
        key for key in keys if key in TUNER_OWNED_KEYS or key.startswith("tune_judge")
    )
    if owned:
        raise ValueError(f"search_space must not override tuner-owned keys: {owned}")
    return [
        Trial({key: value for choice in combo for key, value in choice.items()})
        for combo in itertools.product(*axes)
    ]


def apply_overrides(
    base: Mapping[str, object], overrides: Mapping[str, object]
) -> dict[str, object]:
    """Return a copy of ``base`` with dotted-path ``overrides`` assigned."""
    values = copy.deepcopy(dict(base))
    for path, value in overrides.items():
        *parents, leaf = path.split(".")
        node = values
        for key in parents:
            if node.get(key) is None:
                node[key] = {}
            node = node[key]
        node[leaf] = value
    return values
