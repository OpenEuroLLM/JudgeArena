from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

SACCT_FIELDS = (
    "JobID,JobName%100,Partition,Account,State,ElapsedRaw,Elapsed,"
    "AllocCPUS,AllocNodes,AllocTRES%100,ReqTRES%100"
)
RATE_METRIC_CHOICES = (
    "wall_hours",
    "cpu_hours",
    "gpu_hours",
    "billing_hours",
    "node_hours",
)


@dataclass(frozen=True)
class JobSource:
    job_id: int
    label: str


@dataclass(frozen=True)
class SacctAllocation:
    allocation_id: str
    root_job_id: int
    job_name: str
    partition: str
    account: str
    state: str
    elapsed_seconds: int
    elapsed: str
    alloc_cpus: float
    alloc_nodes: float
    alloc_tres: dict[str, str]
    req_tres: dict[str, str]


@dataclass(frozen=True)
class JobCostSummary:
    job_id: int
    label: str
    partition: str
    account: str
    states: list[str]
    allocation_count: int
    wall_hours: float
    cpu_hours: float
    gpu_hours: float
    billing_hours: float
    node_hours: float
    estimated_cost: float | None = None


def parse_tres_map(tres_spec: str) -> dict[str, str]:
    values: dict[str, str] = {}
    if not tres_spec:
        return values
    for raw_entry in tres_spec.split(","):
        entry = raw_entry.strip()
        if not entry or "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_elapsed_seconds(elapsed: str) -> int:
    if not elapsed:
        return 0
    n_days = 0
    time_part = elapsed
    if "-" in elapsed:
        days_part, time_part = elapsed.split("-", 1)
        n_days = int(days_part)
    hours_str, minutes_str, seconds_str = time_part.split(":")
    return (
        n_days * 86400
        + int(hours_str) * 3600
        + int(minutes_str) * 60
        + int(seconds_str)
    )


def parse_sacct_allocations(sacct_output: str) -> list[SacctAllocation]:
    allocations: list[SacctAllocation] = []
    for raw_line in sacct_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 11:
            raise ValueError(f"Unexpected sacct row with {len(parts)} fields: {line}")
        allocation_id = parts[0].strip()
        root_job_text = allocation_id.split("_", 1)[0]
        allocations.append(
            SacctAllocation(
                allocation_id=allocation_id,
                root_job_id=int(root_job_text),
                job_name=parts[1].strip(),
                partition=parts[2].strip(),
                account=parts[3].strip(),
                state=parts[4].strip(),
                elapsed_seconds=int(parts[5] or "0"),
                elapsed=parts[6].strip(),
                alloc_cpus=float(parts[7] or "0"),
                alloc_nodes=float(parts[8] or "0"),
                alloc_tres=parse_tres_map(parts[9]),
                req_tres=parse_tres_map(parts[10]),
            )
        )
    return allocations


def query_sacct_allocations(job_ids: Iterable[int]) -> list[SacctAllocation]:
    unique_job_ids = [
        str(job_id) for job_id in dict.fromkeys(int(job_id) for job_id in job_ids)
    ]
    if not unique_job_ids:
        return []
    try:
        result = subprocess.run(
            [
                "sacct",
                "-X",
                "--allocations",
                "--parsable2",
                "--noheader",
                f"--format={SACCT_FIELDS}",
                f"--jobs={','.join(unique_job_ids)}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not find `sacct`; run this on a machine with Slurm."
        ) from exc
    if result.returncode != 0:
        message = (
            result.stderr.strip() or result.stdout.strip() or "unknown sacct error"
        )
        raise RuntimeError(f"sacct failed: {message}")
    return parse_sacct_allocations(result.stdout)


def load_job_source_from_path(job_path: str | Path) -> JobSource:
    path = Path(job_path)
    job_dir = path.parent if path.name == "jobid.json" else path
    jobid_path = job_dir / "jobid.json"
    if not jobid_path.is_file():
        raise FileNotFoundError(f"Missing jobid.json in {job_dir}")
    job_id = int(json.loads(jobid_path.read_text())["jobid"])
    metadata_path = job_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        label = str(metadata.get("jobname") or job_dir.name)
    else:
        label = job_dir.name
    return JobSource(job_id=job_id, label=label)


def resolve_job_sources(
    *,
    job_ids: Iterable[int] | None = None,
    job_paths: Iterable[str | Path] | None = None,
) -> list[JobSource]:
    sources: dict[int, JobSource] = {}
    ordered_ids: list[int] = []

    for job_id in job_ids or []:
        normalized_job_id = int(job_id)
        if normalized_job_id in sources:
            continue
        sources[normalized_job_id] = JobSource(
            job_id=normalized_job_id,
            label=str(normalized_job_id),
        )
        ordered_ids.append(normalized_job_id)

    for job_path in job_paths or []:
        source = load_job_source_from_path(job_path)
        if source.job_id not in sources:
            ordered_ids.append(source.job_id)
        sources[source.job_id] = source

    return [sources[job_id] for job_id in ordered_ids]


def _tres_quantity(tres_map: dict[str, str], key: str) -> float:
    raw_value = tres_map.get(key)
    if raw_value is None:
        return 0.0
    numeric_chars: list[str] = []
    for char in raw_value:
        if char.isdigit() or char in {".", "-"}:
            numeric_chars.append(char)
            continue
        break
    numeric_text = "".join(numeric_chars)
    return float(numeric_text) if numeric_text else 0.0


def summarize_job_costs(
    sources: list[JobSource],
    allocations: list[SacctAllocation],
    *,
    rate_metric: str | None = None,
    hourly_rate: float | None = None,
) -> list[JobCostSummary]:
    allocations_by_job_id: dict[int, list[SacctAllocation]] = {
        source.job_id: [] for source in sources
    }
    for allocation in allocations:
        if allocation.root_job_id in allocations_by_job_id:
            allocations_by_job_id[allocation.root_job_id].append(allocation)

    missing_job_ids = [
        str(source.job_id)
        for source in sources
        if not allocations_by_job_id[source.job_id]
    ]
    if missing_job_ids:
        raise RuntimeError(
            "No sacct allocation rows returned for job IDs: "
            + ", ".join(missing_job_ids)
        )

    summaries: list[JobCostSummary] = []
    for source in sources:
        job_allocations = allocations_by_job_id[source.job_id]
        wall_hours = sum(row.elapsed_seconds for row in job_allocations) / 3600.0
        cpu_hours = (
            sum(row.elapsed_seconds * row.alloc_cpus for row in job_allocations)
            / 3600.0
        )
        gpu_hours = (
            sum(
                row.elapsed_seconds * _tres_quantity(row.alloc_tres, "gres/gpu")
                for row in job_allocations
            )
            / 3600.0
        )
        billing_hours = (
            sum(
                row.elapsed_seconds * _tres_quantity(row.alloc_tres, "billing")
                for row in job_allocations
            )
            / 3600.0
        )
        node_hours = (
            sum(row.elapsed_seconds * row.alloc_nodes for row in job_allocations)
            / 3600.0
        )
        metric_value = (
            _summary_metric_value(
                wall_hours=wall_hours,
                cpu_hours=cpu_hours,
                gpu_hours=gpu_hours,
                billing_hours=billing_hours,
                node_hours=node_hours,
                rate_metric=rate_metric,
            )
            if hourly_rate is not None
            else None
        )
        summaries.append(
            JobCostSummary(
                job_id=source.job_id,
                label=source.label,
                partition=",".join(
                    sorted({row.partition for row in job_allocations if row.partition})
                ),
                account=",".join(
                    sorted({row.account for row in job_allocations if row.account})
                ),
                states=sorted({row.state for row in job_allocations if row.state}),
                allocation_count=len(job_allocations),
                wall_hours=wall_hours,
                cpu_hours=cpu_hours,
                gpu_hours=gpu_hours,
                billing_hours=billing_hours,
                node_hours=node_hours,
                estimated_cost=(
                    metric_value * hourly_rate
                    if metric_value is not None and hourly_rate is not None
                    else None
                ),
            )
        )
    return summaries


def _summary_metric_value(
    *,
    wall_hours: float,
    cpu_hours: float,
    gpu_hours: float,
    billing_hours: float,
    node_hours: float,
    rate_metric: str | None,
) -> float:
    metrics = {
        "wall_hours": wall_hours,
        "cpu_hours": cpu_hours,
        "gpu_hours": gpu_hours,
        "billing_hours": billing_hours,
        "node_hours": node_hours,
    }
    if rate_metric is None:
        raise ValueError("rate_metric must be set when hourly_rate is provided")
    return metrics[rate_metric]


def total_summary(
    summaries: list[JobCostSummary], *, hourly_rate: float | None = None
) -> JobCostSummary:
    return JobCostSummary(
        job_id=0,
        label="TOTAL",
        partition=",".join(
            sorted({summary.partition for summary in summaries if summary.partition})
        ),
        account=",".join(
            sorted({summary.account for summary in summaries if summary.account})
        ),
        states=sorted({state for summary in summaries for state in summary.states}),
        allocation_count=sum(summary.allocation_count for summary in summaries),
        wall_hours=sum(summary.wall_hours for summary in summaries),
        cpu_hours=sum(summary.cpu_hours for summary in summaries),
        gpu_hours=sum(summary.gpu_hours for summary in summaries),
        billing_hours=sum(summary.billing_hours for summary in summaries),
        node_hours=sum(summary.node_hours for summary in summaries),
        estimated_cost=(
            sum(summary.estimated_cost or 0.0 for summary in summaries)
            if hourly_rate is not None
            else None
        ),
    )


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _format_cost(value: float, currency: str) -> str:
    return f"{currency} {value:.2f}"


def _tabular_rows(
    summaries: list[JobCostSummary], *, currency: str, include_cost: bool
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in summaries:
        row = {
            "job": summary.label,
            "job_id": str(summary.job_id),
            "tasks": str(summary.allocation_count),
            "state": ",".join(summary.states),
            "gpu_h": _format_float(summary.gpu_hours),
            "billing_h": _format_float(summary.billing_hours),
            "cpu_h": _format_float(summary.cpu_hours),
            "wall_h": _format_float(summary.wall_hours),
        }
        if include_cost and summary.estimated_cost is not None:
            row["cost"] = _format_cost(summary.estimated_cost, currency)
        rows.append(row)
    return rows


def render_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))
    header_line = "  ".join(f"{header:<{widths[header]}}" for header in headers)
    separator_line = "  ".join("-" * widths[header] for header in headers)
    row_lines = [
        "  ".join(f"{row[header]:<{widths[header]}}" for header in headers)
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def _summary_to_dict(summary: JobCostSummary) -> dict[str, object]:
    return asdict(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m judgearena.slurm_costs",
        description="Summarize Slurm job usage and simple cost estimates.",
    )
    parser.add_argument(
        "--job-id",
        action="append",
        type=int,
        default=[],
        help="Root Slurm job ID to summarize. Repeatable.",
    )
    parser.add_argument(
        "--job-path",
        action="append",
        default=[],
        help="Path to a slurmpilot job directory or its jobid.json. Repeatable.",
    )
    parser.add_argument(
        "--rate-metric",
        choices=RATE_METRIC_CHOICES,
        default="gpu_hours",
        help="Metric used for the optional hourly rate conversion.",
    )
    parser.add_argument(
        "--hourly-rate",
        type=float,
        default=None,
        help="Optional hourly rate applied to --rate-metric.",
    )
    parser.add_argument(
        "--currency",
        default="EUR",
        help="Currency label for the optional cost estimate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a text table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = resolve_job_sources(job_ids=args.job_id, job_paths=args.job_path)
    if not sources:
        raise SystemExit("Provide at least one --job-id or --job-path.")
    allocations = query_sacct_allocations(source.job_id for source in sources)
    summaries = summarize_job_costs(
        sources,
        allocations,
        rate_metric=args.rate_metric,
        hourly_rate=args.hourly_rate,
    )
    total = total_summary(summaries, hourly_rate=args.hourly_rate)
    if args.json:
        payload = {
            "jobs": [_summary_to_dict(summary) for summary in summaries],
            "total": _summary_to_dict(total),
            "rate_metric": args.rate_metric if args.hourly_rate is not None else None,
            "hourly_rate": args.hourly_rate,
            "currency": args.currency if args.hourly_rate is not None else None,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    table_rows = _tabular_rows(
        [*summaries, total],
        currency=args.currency,
        include_cost=args.hourly_rate is not None,
    )
    print(render_table(table_rows))
    if args.hourly_rate is None:
        print(
            "\nNo hourly rate was provided. Pass --hourly-rate together with "
            "--rate-metric to convert one of the reported hour metrics into money."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
