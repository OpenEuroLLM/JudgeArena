"""Construction and execution of metrics for annotations and battles."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from inspect import Parameter, signature
from typing import Protocol

import numpy as np
import pandas as pd

from judgearena.benchmarks.elo.scoring import BradleyTerryMetric
from judgearena.benchmarks.meta_eval.scoring import (
    MetaEvalAgreementMetric,
    MetaEvalEloGapMetric,
    MetaEvalRankingMetric,
)
from judgearena.benchmarks.pairwise.scoring.alpaca_eval import (
    AlpacaEvalLengthControlledMetric,
)
from judgearena.benchmarks.pairwise.scoring.arena_hard import (
    ArenaHardV01Metric,
    ArenaHardV20Metric,
)
from judgearena.benchmarks.pairwise.scoring.metrics import (
    AbsoluteScoreMetric,
    LengthControlledWinrateMetric,
    PairwiseWinRateMetric,
)
from judgearena.log import get_logger

logger = get_logger(__name__)


class _Metric(Protocol):
    """The calculation and renderer supplied by every metric implementation."""

    def calculate(
        self, battles: pd.DataFrame, **runtime: object
    ) -> dict[str, object]: ...

    @staticmethod
    def render(result: dict[str, object]) -> str: ...


class MetricRequest(Protocol):
    """The request fields needed to construct and group one metric."""

    metric: str
    breakdown_by: tuple[str, ...]
    parameters: Mapping[str, object]


_METRIC_TYPES: dict[str, type[_Metric]] = {
    "pairwise_win_rate": PairwiseWinRateMetric,
    "length_controlled_winrate": LengthControlledWinrateMetric,
    "bradley_terry": BradleyTerryMetric,
    "arena_hard_v01": ArenaHardV01Metric,
    "arena_hard_v20": ArenaHardV20Metric,
    "alpaca_eval_length_controlled": AlpacaEvalLengthControlledMetric,
    "meta_eval_agreement": MetaEvalAgreementMetric,
    "meta_eval_elo_gap": MetaEvalEloGapMetric,
    "meta_eval_ranking": MetaEvalRankingMetric,
    "absolute_score": AbsoluteScoreMetric,
}

ConfiguredMetrics = tuple[tuple[MetricRequest, _Metric | Exception], ...]


def register_metric[T: type[_Metric]](name: str):
    """Register a metric class for task/config use in this Python process.

    ``calculate(rows)`` consumes canonical annotations/outcomes. Optional
    ``required_columns`` lists its input fields; failures are saved as ``error``.
    Pairwise rows have model_a/model_b/pref; independent rows have model/parsed.
    Scalar score and candidate score_A/score_B depend on the selected parser.
    Optional runtime keywords include rng and references: pairwise baselines,
    Elo anchor models, or the selected meta-evaluation model pool, respectively.
    """

    def register(metric: T) -> T:
        if not name or name in _METRIC_TYPES:
            raise ValueError(f"Metric {name!r} is empty or already registered.")
        _METRIC_TYPES[name] = metric
        return metric

    return register


def available_metrics() -> tuple[str, ...]:
    """Return the stable set of supported metric identifiers."""
    return tuple(sorted(_METRIC_TYPES))


def _metric_type(name: str) -> type[_Metric]:
    try:
        return _METRIC_TYPES[name]
    except KeyError as exc:
        choices = ", ".join(available_metrics())
        raise ValueError(
            f"Unknown metric {name!r}; available metrics: {choices}."
        ) from exc


def build_metric(name: str, parameters: Mapping[str, object] | None = None) -> _Metric:
    """Build one fresh, configured metric instance."""
    try:
        return _metric_type(name)(**dict(parameters or {}))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid parameters for metric {name!r}: {exc}") from exc


def build_metrics(
    requests: Sequence[MetricRequest],
    *,
    parameter_overrides_by_metric: Mapping[str, Mapping[str, object]] | None = None,
) -> ConfiguredMetrics:
    """Build metrics in order, retaining construction failures for the report."""
    overrides = parameter_overrides_by_metric or {}
    configured: list[tuple[MetricRequest, _Metric | Exception]] = []
    for request in requests:
        parameters = dict(request.parameters)
        parameters.update(overrides.get(request.metric, {}))
        try:
            configured.append((request, build_metric(request.metric, parameters)))
        except Exception as exc:
            logger.warning("Skipping metric %r: %s", request.metric, exc)
            configured.append((request, exc))
    return tuple(configured)


def _group_value(value: object) -> object:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def calculate_metrics(
    battles: pd.DataFrame,
    metrics: ConfiguredMetrics,
    *,
    runtime: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Supply accepted runtime keywords and record failures without losing annotations.

    An integer rng gives each metric a stream derived from its name and seed.
    A Generator is copied at its current state per metric; breakdowns continue
    that metric's stream. Other values, such as references, pass through unchanged.
    """
    runtime = runtime or {}
    calculated: dict[str, dict[str, object]] = {}
    for request, metric in metrics:
        try:
            if isinstance(metric, Exception):
                raise metric
            parameters = signature(metric.calculate).parameters
            accepts_kwargs = any(
                p.kind == Parameter.VAR_KEYWORD for p in parameters.values()
            )
            kwargs = {
                key: value
                for key, value in runtime.items()
                if accepts_kwargs
                or (
                    key in parameters
                    and parameters[key].kind
                    in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
                )
            }
            if "rng" in kwargs:
                rng = kwargs["rng"]
                if isinstance(rng, int):
                    digest = hashlib.sha256(
                        f"{rng}\0{request.metric}".encode()
                    ).digest()
                    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
                else:
                    rng = deepcopy(rng)
                kwargs["rng"] = rng
            missing = set(getattr(metric, "required_columns", ())) - set(
                battles.columns
            )
            if missing:
                raise ValueError(f"Missing metric columns: {sorted(missing)}")
            values = dict(metric.calculate(battles, **kwargs))
        except Exception as exc:
            logger.warning("Skipping metric %r: %s", request.metric, exc)
            calculated[request.metric] = {"error": str(exc)}
            continue
        if request.breakdown_by:
            groups: dict[str, list[dict[str, object]]] = {}
            for field in request.breakdown_by:
                try:
                    groups[field] = [
                        {
                            "group": _group_value(key),
                            "values": metric.calculate(group, **kwargs),
                        }
                        for key, group in battles.groupby(
                            field, sort=False, dropna=False
                        )
                    ]
                except Exception as exc:
                    values.setdefault("breakdown_errors", {})[field] = str(exc)
                    logger.warning(
                        "Skipping metric %r breakdown %r: %s",
                        request.metric,
                        field,
                        exc,
                    )
            values["groups"] = groups
        calculated[request.metric] = values
    return calculated


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def render_metrics(results: Mapping[str, dict[str, object]]) -> str:
    """Render metric results in their configured dictionary order."""
    sections: list[str] = []
    for name, result in results.items():
        if "error" in result:
            sections.append(f"{name}: failed — {result['error']}")
            continue
        try:
            metric_type = _metric_type(name)
            overall = {
                key: value
                for key, value in result.items()
                if key not in ("groups", "breakdown_errors")
            }
            sections.append(
                metric_type.render(overall) if overall else f"{name}: unavailable"
            )
            for field, error in result.get("breakdown_errors", {}).items():
                sections.append(f"{field}: failed — {error}")
            for field, groups in result.get("groups", {}).items():
                for group in groups:
                    rendered = (
                        metric_type.render(group["values"])
                        if group["values"]
                        else f"{name}: unavailable"
                    )
                    sections.append(f"{field}={group['group']}:\n{_indent(rendered)}")
        except Exception as exc:
            logger.warning("Skipping rendering for metric %r: %s", name, exc)
    return "\n\n".join(sections)
