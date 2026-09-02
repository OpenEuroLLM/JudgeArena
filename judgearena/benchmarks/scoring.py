"""Construction and execution of configured battle metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import pandas as pd

from judgearena.benchmarks.elo.scoring import BradleyTerryMetric
from judgearena.benchmarks.pairwise.scoring.metrics import (
    LengthControlledWinrateMetric,
    PairwiseWinRateMetric,
)


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
    group_by: tuple[str, ...]
    parameters: Mapping[str, object]


_METRIC_TYPES: dict[str, type[_Metric]] = {
    "pairwise_win_rate": PairwiseWinRateMetric,
    "length_controlled_winrate": LengthControlledWinrateMetric,
    "bradley_terry": BradleyTerryMetric,
}

ConfiguredMetrics = tuple[tuple[MetricRequest, _Metric], ...]


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
    """Build requested metrics in order, applying temporary runtime overrides."""
    overrides = parameter_overrides_by_metric or {}
    configured: list[tuple[MetricRequest, _Metric]] = []
    for request in requests:
        parameters = dict(request.parameters)
        parameters.update(overrides.get(request.metric, {}))
        configured.append((request, build_metric(request.metric, parameters)))
    return tuple(configured)


def _group_value(value: object) -> object:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def calculate_metrics(
    battles: pd.DataFrame,
    metrics: ConfiguredMetrics,
    *,
    runtime_by_metric: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Run configured metrics over one battle table."""
    runtime_by_metric = runtime_by_metric or {}
    calculated: dict[str, dict[str, object]] = {}
    for request, metric in metrics:
        runtime = dict(runtime_by_metric.get(request.metric, {}))
        values = dict(metric.calculate(battles, **runtime))
        if request.group_by:
            groups: dict[str, list[dict[str, object]]] = {}
            for field in request.group_by:
                if field not in battles:
                    raise ValueError(
                        f"Metric {request.metric!r} cannot group by missing column "
                        f"{field!r}."
                    )
                groups[field] = [
                    {
                        "group": _group_value(key),
                        "values": metric.calculate(group, **runtime),
                    }
                    for key, group in battles.groupby(field, sort=False, dropna=False)
                ]
            values["groups"] = groups
        calculated[request.metric] = values
    return calculated


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def render_metrics(results: Mapping[str, dict[str, object]]) -> str:
    """Render metric results in their configured dictionary order."""
    sections: list[str] = []
    for name, result in results.items():
        metric_type = _metric_type(name)
        overall = {key: value for key, value in result.items() if key != "groups"}
        sections.append(metric_type.render(overall))
        for field, groups in result.get("groups", {}).items():
            for group in groups:
                rendered = metric_type.render(group["values"])
                sections.append(f"{field}={group['group']}:\n{_indent(rendered)}")
    return "\n\n".join(sections)
