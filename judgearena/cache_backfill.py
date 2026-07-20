"""Backfill hosted judge inference rows from saved run folders into the unified cache."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from judgearena.cache_backfill_discovery import (
    SKIP_REASON_BY_KIND,
    ArtifactKind,
    ClassifiedSource,
    discover_sources,
)
from judgearena.cache_backfill_sources import (
    BackfillRow,
    SourceExtraction,
    extract_gae_rows,
    extract_meta_eval_rows,
    extract_mt_bench_rows,
)
from judgearena.log import get_logger
from judgearena.store_sqlite import (
    INFERENCE_DB_NAME,
    SQLiteInferenceStore,
    descriptor_hash,
    stable_json_dumps,
    store_folder,
    write_store_metadata,
)

logger = get_logger(__name__)

BACKFILL_PUSHED_BY = "backfill"


@dataclass
class BackfillReport:
    written: int = 0
    existing: int = 0
    rows_planned: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    sources: dict[str, dict[str, int]] = field(default_factory=dict)
    runs_processed: int = 0
    dry_run: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "existing": self.existing,
            "rows_planned": self.rows_planned,
            "skipped": dict(sorted(self.skipped.items())),
            "sources": self.sources,
            "runs_processed": self.runs_processed,
            "dry_run": self.dry_run,
        }


def _merge_skip_counts(target: dict[str, int], delta: dict[str, int]) -> None:
    for reason, count in delta.items():
        target[reason] = target.get(reason, 0) + count


def _extract_rows(classified: ClassifiedSource) -> SourceExtraction:
    run_dir = classified.run_dir
    assert run_dir is not None
    if classified.kind == ArtifactKind.GAE_RUN:
        return extract_gae_rows(run_dir)
    if classified.kind == ArtifactKind.MT_BENCH_RUN:
        return extract_mt_bench_rows(run_dir)
    if classified.kind == ArtifactKind.META_EVAL_RUN:
        return extract_meta_eval_rows(run_dir)
    raise ValueError(f"Unsupported migratable kind: {classified.kind}")


def _row_cache_key(row: BackfillRow) -> tuple[str, str, str, str]:
    input_hash = descriptor_hash(row.canonical_input, length=None)
    return row.task, row.model_spec, descriptor_hash(row.descriptor), input_hash


def _dedupe_and_detect_conflicts(
    rows: list[BackfillRow],
) -> tuple[list[BackfillRow], int]:
    grouped: dict[tuple[str, str, str, str], list[BackfillRow]] = defaultdict(list)
    for row in rows:
        grouped[_row_cache_key(row)].append(row)

    final: list[BackfillRow] = []
    dropped_rows = 0
    for key_rows in grouped.values():
        if len({row.output_text for row in key_rows}) > 1:
            dropped_rows += len(key_rows)
            continue
        # Preserve duplicates so each distinct metadata association is written.
        final.extend(key_rows)

    return final, dropped_rows


def _classify_rows_for_cell(
    cell_rows: list[BackfillRow],
    stored_outputs: dict[str, str],
) -> tuple[list[BackfillRow], list[BackfillRow], list[BackfillRow], int]:
    to_write: list[BackfillRow] = []
    metadata_rows: list[BackfillRow] = []
    conflicting: list[BackfillRow] = []
    existing_count = 0

    rows_by_hash: dict[str, list[BackfillRow]] = defaultdict(list)
    for row in cell_rows:
        rows_by_hash[descriptor_hash(row.canonical_input, length=None)].append(row)

    for input_hash, hash_rows in rows_by_hash.items():
        stored_output = stored_outputs.get(input_hash)
        if stored_output is None:
            to_write.append(hash_rows[0])
            metadata_rows.extend(hash_rows)
            continue
        if stored_output == hash_rows[0].output_text:
            existing_count += 1
            metadata_rows.extend(hash_rows)
            continue
        conflicting.extend(hash_rows)

    return to_write, metadata_rows, conflicting, existing_count


def _write_rows(
    rows: list[BackfillRow],
    store_root: Path,
    *,
    dry_run: bool,
) -> tuple[int, int, dict[str, int]]:
    if not rows:
        return 0, 0, {}

    by_cell: dict[tuple[str, str, str], list[BackfillRow]] = defaultdict(list)
    for row in rows:
        by_cell[(row.task, row.model_spec, descriptor_hash(row.descriptor))].append(row)

    written = 0
    existing = 0
    skipped: dict[str, int] = {}
    run_id = str(uuid.uuid4())

    for (task, model_spec, config_hash), cell_rows in by_cell.items():
        descriptor = cell_rows[0].descriptor
        try:
            folder = store_folder(store_root, task, model_spec, config_hash)
            db_path = folder / INFERENCE_DB_NAME
            input_hashes = [
                descriptor_hash(row.canonical_input, length=None) for row in cell_rows
            ]
            unique_hashes = list(dict.fromkeys(input_hashes))

            if dry_run:
                if db_path.exists():
                    with SQLiteInferenceStore(db_path, readonly=True) as store:
                        stored_outputs = store.outputs_by_hash(unique_hashes)
                        to_write, _, conflicting, existing_count = (
                            _classify_rows_for_cell(cell_rows, stored_outputs)
                        )
                        existing += existing_count
                        written += len(to_write)
                        if conflicting:
                            _merge_skip_counts(
                                skipped,
                                {"conflicting_existing_output": len(conflicting)},
                            )
                else:
                    written += len(unique_hashes)
                continue

            write_store_metadata(folder, descriptor)
            with SQLiteInferenceStore(db_path) as store:
                stored_outputs = store.outputs_by_hash(unique_hashes)
                to_write, metadata_rows, conflicting, existing_count = (
                    _classify_rows_for_cell(cell_rows, stored_outputs)
                )
                existing += existing_count
                if conflicting:
                    _merge_skip_counts(
                        skipped,
                        {"conflicting_existing_output": len(conflicting)},
                    )

                outputs_payload = []
                for row in to_write:
                    input_hash = descriptor_hash(row.canonical_input, length=None)
                    outputs_payload.append(
                        {
                            "input_hash": input_hash,
                            "input_text": row.canonical_input,
                            "output_text": row.output_text,
                            "producer_metadata_json": stable_json_dumps(
                                row.producer_metadata
                            ),
                        }
                    )

                metadata_payload = []
                for row in metadata_rows:
                    metadata_payload.append(
                        {
                            "input_hash": descriptor_hash(
                                row.canonical_input, length=None
                            ),
                            "metadata_json": stable_json_dumps(row.row_metadata),
                        }
                    )

                if outputs_payload and metadata_payload:
                    outputs_written, _ = store.save_outputs_and_metadata(
                        pd.DataFrame(outputs_payload),
                        pd.DataFrame(metadata_payload),
                        pushed_by=BACKFILL_PUSHED_BY,
                        run_id=run_id,
                        replace=False,
                    )
                    written += outputs_written
                elif outputs_payload:
                    written += store.save_outputs(
                        pd.DataFrame(outputs_payload),
                        pushed_by=BACKFILL_PUSHED_BY,
                        run_id=run_id,
                        replace=False,
                    )
                elif metadata_payload:
                    store.save_metadata(pd.DataFrame(metadata_payload), run_id=run_id)
        except (OSError, ValueError, sqlite3.Error) as exc:
            logger.warning(
                "Cell integrity error for task=%s model=%s config=%s: %s",
                task,
                model_spec,
                config_hash,
                exc,
            )
            _merge_skip_counts(skipped, {"cell_integrity_error": len(cell_rows)})

    return written, existing, skipped


def backfill_sources(
    sources: list[Path | str],
    store_root: Path | str,
    *,
    dry_run: bool = False,
) -> BackfillReport:
    """Discover saved judge runs and insert reconstructable rows into the store."""
    report = BackfillReport(dry_run=dry_run)
    resolved_sources = [Path(source) for source in sources]
    discovery = discover_sources(resolved_sources)

    for skipped in discovery.skipped:
        reason = SKIP_REASON_BY_KIND.get(skipped.kind, "unknown_source")
        report.skipped[reason] = report.skipped.get(reason, 0) + 1

    extracted_rows: list[BackfillRow] = []
    for classified in discovery.migratable_runs:
        run_dir = classified.run_dir
        try:
            extraction = _extract_rows(classified)
        except Exception as exc:
            logger.warning(
                "Source extraction failed for %s: %s",
                run_dir.name if run_dir else classified.path,
                exc,
            )
            _merge_skip_counts(report.skipped, {"source_extraction_failed": 1})
            continue

        _merge_skip_counts(report.skipped, extraction.skipped)
        source_stats = report.sources.setdefault(
            extraction.source_kind,
            {"runs": 0, "rows_extracted": 0},
        )
        source_stats["runs"] += 1
        source_stats["rows_extracted"] += len(extraction.rows)
        extracted_rows.extend(extraction.rows)
        report.runs_processed += 1

    deduped_rows, conflict_count = _dedupe_and_detect_conflicts(extracted_rows)
    report.rows_planned = len(deduped_rows)
    if conflict_count:
        _merge_skip_counts(report.skipped, {"conflicting_outputs": conflict_count})

    written, existing, write_skipped = _write_rows(
        deduped_rows,
        Path(store_root).expanduser(),
        dry_run=dry_run,
    )
    _merge_skip_counts(report.skipped, write_skipped)
    report.written = written
    report.existing = existing
    return report


def write_report(report: BackfillReport, path: Path | str) -> None:
    """Persist a JSON-safe backfill report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_jsonable(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def log_report_summary(report: BackfillReport) -> None:
    logger.info(
        "Backfill complete: written=%d existing=%d rows_planned=%d "
        "runs=%d dry_run=%s skipped=%s",
        report.written,
        report.existing,
        report.rows_planned,
        report.runs_processed,
        report.dry_run,
        report.skipped,
    )
