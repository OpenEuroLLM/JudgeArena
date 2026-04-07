import json

import pytest

from judgearena.slurm_costs import (
    JobSource,
    _tres_quantity,
    load_job_source_from_path,
    parse_elapsed_seconds,
    parse_sacct_allocations,
    parse_tres_map,
    resolve_job_sources,
    summarize_job_costs,
    total_summary,
)


def test_parse_tres_map_preserves_entries_and_extracts_numeric_quantities():
    tres_map = parse_tres_map("billing=2,cpu=2,gres/gpu=1,mem=125G,node=1")

    assert tres_map["mem"] == "125G"
    assert _tres_quantity(tres_map, "billing") == 2.0
    assert _tres_quantity(tres_map, "gres/gpu") == 1.0


def test_parse_elapsed_seconds_supports_day_prefix():
    assert parse_elapsed_seconds("1-02:03:04") == 93784


def test_load_job_source_from_path_uses_metadata_jobname(tmp_path):
    job_dir = tmp_path / "bench" / "alpaca-eval-2026-04-06-16-25-10"
    job_dir.mkdir(parents=True)
    (job_dir / "jobid.json").write_text(json.dumps({"jobid": 28707665}))
    (job_dir / "metadata.json").write_text(
        json.dumps({"jobname": "bench/alpaca-eval-2026-04-06-16-25-10"})
    )

    source = load_job_source_from_path(job_dir)
    resolved = resolve_job_sources(job_ids=[28707665], job_paths=[job_dir])

    assert source == JobSource(
        job_id=28707665,
        label="bench/alpaca-eval-2026-04-06-16-25-10",
    )
    assert resolved == [source]


def test_summarize_job_costs_aggregates_job_arrays_and_rate_conversion():
    sacct_output = "\n".join(
        [
            "28707665_0|bench/alpaca-eval|mldlc2_gpu-l40s|ml-dlc2|COMPLETED|60|00:01:00|2|1|billing=2,cpu=2,gres/gpu=1,node=1|billing=2,cpu=2,gres/gpu=1,node=1",
            "28707665_1|bench/alpaca-eval|mldlc2_gpu-l40s|ml-dlc2|COMPLETED|90|00:01:30|2|1|billing=2,cpu=2,gres/gpu=1,node=1|billing=2,cpu=2,gres/gpu=1,node=1",
            "28708344_0|bench/arena-hard|mldlc2_gpu-l40s|ml-dlc2|COMPLETED|120|00:02:00|2|1|billing=2,cpu=2,gres/gpu=1,node=1|billing=2,cpu=2,gres/gpu=1,node=1",
        ]
    )
    allocations = parse_sacct_allocations(sacct_output)
    sources = [
        JobSource(job_id=28707665, label="bench/alpaca-eval"),
        JobSource(job_id=28708344, label="bench/arena-hard"),
    ]

    summaries = summarize_job_costs(
        sources,
        allocations,
        rate_metric="gpu_hours",
        hourly_rate=3.5,
    )
    total = total_summary(summaries, hourly_rate=3.5)

    assert summaries[0].allocation_count == 2
    assert summaries[0].gpu_hours == pytest.approx(150 / 3600)
    assert summaries[0].billing_hours == pytest.approx(300 / 3600)
    assert summaries[0].estimated_cost == pytest.approx((150 / 3600) * 3.5)
    assert summaries[1].gpu_hours == pytest.approx(120 / 3600)
    assert total.gpu_hours == pytest.approx(270 / 3600)
    assert total.cpu_hours == pytest.approx(540 / 3600)
    assert total.estimated_cost == pytest.approx((270 / 3600) * 3.5)
