"""Phase B serial runner: judge a frozen (dataset x model) cell matrix.

Submitted as a single Slurm job by ``slurmpilot_scripts/launch_phase_b_serial.py``.
For each cell:

    1. Read pre-frozen CLI args from ``phase_b_cells.json`` (or the override
       named by ``JUDGEARENA_PHASE_B_CELLS_FILENAME``) written by the launcher
       at submit time on the login node, where the launcher module is
       importable. Freezing args at submit time makes the run reproducible: a
       launcher edit landed mid-run does not change what the in-flight job
       executes.
    2. Skip if a JSON-valid ``results-*.json`` for that cell already exists
       in ``$CWD/results/<name>/`` (mirrors ``arena.py``'s skip semantics
       plus a JSON-validity check so a half-written file doesn't masquerade
       as done).
    3. Otherwise spawn ``python -m judgearena.cli`` as a
       subprocess. Subprocess isolation matters because (a) vLLM's CUDA
       context lives in the subprocess so a CUDA fault doesn't poison the
       next cell, (b) one cell's transient OpenRouter outage doesn't lose
       the other 47 cells' work.

Continues across failures and prints a per-cell FAIL/SKIP/OK summary at the
end. Exits non-zero if any cell errored, so Slurm marks the job FAILED.

Re-running is idempotent: cells with a valid results-*.json are skipped, so
a partial run can resume cleanly from where it left off.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from judgearena.generate_and_evaluate import CliArgs

DEFAULT_CELLS_FILENAME = "phase_b_cells.json"
SUPPORTED_CLI_KEYS = frozenset(
    (set(CliArgs.__dataclass_fields__.keys()) - {"task"}) | {"dataset"}
)


def _slugify(value: str) -> str:
    """Mirror the ``"/" -> "_"`` substitution in ``generate_and_evaluate``'s
    output-folder naming."""
    return value.replace("/", "_")


def _validate_results_file(path: Path) -> tuple[bool, str]:
    """Two-part check that ``path`` represents a usable Phase B cell.

    1. JSON parses (catches truncated writes, IO errors).
    2. ``num_missing < num_battles`` (catches all-missing runs where every
       judge call returned a non-parseable verdict — the subprocess exits
       0 in this case but the file is operationally useless).

    The schema is the flat dict written by ``generate_and_evaluate`` at
    ``[generate_and_evaluate.py:556-569]``; we still defensively unwrap a
    nested ``summary`` shape because previous runs / future refactors may
    differ.
    """
    try:
        with path.open() as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"json_invalid:{exc.__class__.__name__}"

    summary = payload.get("summary") or payload
    num_missing = summary.get("num_missing")
    num_battles = summary.get("num_battles")
    if num_battles is None:
        num_battles = summary.get("n")
    if num_missing is None or num_battles is None:
        return False, "missing_num_missing_or_num_battles"
    if int(num_battles) <= 0:
        return False, "num_battles_zero"
    if int(num_missing) >= int(num_battles):
        return (
            False,
            f"all_missing:num_missing={num_missing}/{num_battles}",
        )
    return True, ""


def _find_valid_results_path(
    *,
    result_folder: Path,
    dataset: str,
    model_a: str,
    judge_model: str,
    swap_mode: str,
) -> Path | None:
    """Return the most recent valid ``results-*.json`` for this cell if any.

    The folder name embeds ``baseline_plan.display_name`` which is resolved at
    runtime (and varies for arena-hard's per-row baseline plans). We therefore
    glob over the baseline slot rather than recomputing the resolution here.
    The trailing ``*`` after ``swap_mode`` covers MT-Bench's ``-mtbench`` suffix
    appended by ``_build_mt_bench_result_name`` ([mt_bench_utils.py:176-182]);
    without it MT-Bench cells were misclassified as FAIL even when results were
    written correctly.
    Picks the newest mtime when multiple folders match (e.g. baseline string
    changed between runs) so re-validation after a re-judge sees the fresh
    output, not a stale one.
    """
    pattern = f"{dataset}-{_slugify(model_a)}-*-{_slugify(judge_model)}-{swap_mode}*"
    candidates: list[Path] = []
    for cell_folder in result_folder.glob(pattern):
        candidates.extend(cell_folder.glob("results-*.json"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for results_path in candidates:
        is_valid, reason = _validate_results_file(results_path)
        if is_valid:
            return results_path
        print(
            f"  [warn] {results_path} exists but failed validity check "
            f"({reason}); ignoring for skip-detection.",
            flush=True,
        )
    return None


def _args_to_argv(args: dict[str, object]) -> list[str]:
    """Convert ``build_eval_args`` output to argv flags accepted by
    ``judgearena.cli``.

    Most booleans use the ``--flag=true/false`` form so they go through
    ``parse_optional_bool`` inside the shared CLI argument helpers. The top-level
    ``judgearena.cli`` parser handles ``--use_tqdm`` specially as a plain
    ``store_true`` flag, so emit it only when truthy.

    Frozen cells can include metadata used only for reporting. Drop anything
    that is not a real CLI flag so the subprocess argv stays parseable.
    """
    argv: list[str] = []
    for key, value in args.items():
        if key not in SUPPORTED_CLI_KEYS:
            continue
        if value is None:
            continue
        if key == "dataset":
            argv.append(f"--task={value}")
            continue
        if key == "use_tqdm":
            if value:
                argv.append("--use_tqdm")
            continue
        if isinstance(value, bool):
            argv.append(f"--{key}={'true' if value else 'false'}")
        else:
            argv.append(f"--{key}={value}")
    return argv


@dataclass
class CellOutcome:
    dataset: str
    model_a: str
    status: str
    elapsed_seconds: float
    note: str = ""


def _run_cell(
    *,
    dataset: str,
    model_a: str,
    judge_model: str,
    swap_mode: str,
    args: dict[str, object],
    result_folder: Path,
) -> CellOutcome:
    started_at = time.time()
    existing = _find_valid_results_path(
        result_folder=result_folder,
        dataset=dataset,
        model_a=model_a,
        judge_model=judge_model,
        swap_mode=swap_mode,
    )
    if existing is not None:
        return CellOutcome(
            dataset=dataset,
            model_a=model_a,
            status="SKIP",
            elapsed_seconds=time.time() - started_at,
            note=f"reusing {existing.relative_to(result_folder)}",
        )

    cell_args = dict(args)
    cell_args["result_folder"] = str(result_folder)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "judgearena.cli",
        *_args_to_argv(cell_args),
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - started_at
    if proc.returncode != 0:
        return CellOutcome(
            dataset=dataset,
            model_a=model_a,
            status="FAIL",
            elapsed_seconds=elapsed,
            note=f"exit_code={proc.returncode}",
        )

    # Subprocess isolation means we can't observe ``num_missing`` inside the
    # child; re-validate the on-disk artefact here so a child that exits 0
    # while writing an all-missing file (e.g. every OpenRouter call returned
    # a non-parseable verdict) is surfaced as FAIL rather than silently OK.
    written = _find_valid_results_path(
        result_folder=result_folder,
        dataset=dataset,
        model_a=model_a,
        judge_model=judge_model,
        swap_mode=swap_mode,
    )
    if written is None:
        pattern = (
            f"{dataset}-{_slugify(model_a)}-*-{_slugify(judge_model)}-{swap_mode}*"
        )
        any_file = next(
            (p for d in result_folder.glob(pattern) for p in d.glob("results-*.json")),
            None,
        )
        if any_file is None:
            note = "exit_code=0 but no results-*.json was written"
        else:
            _, reason = _validate_results_file(any_file)
            note = f"exit_code=0 but {any_file.name} failed validity ({reason})"
        return CellOutcome(
            dataset=dataset,
            model_a=model_a,
            status="FAIL",
            elapsed_seconds=elapsed,
            note=note,
        )

    return CellOutcome(
        dataset=dataset,
        model_a=model_a,
        status="OK",
        elapsed_seconds=elapsed,
        note=f"wrote {written.relative_to(result_folder)}",
    )


def _format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 90:
        return f"{minutes}m{sec:02d}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins:02d}m"


def _cells_path(script_dir: Path) -> Path:
    filename = os.getenv(
        "JUDGEARENA_PHASE_B_CELLS_FILENAME",
        DEFAULT_CELLS_FILENAME,
    )
    path = Path(filename)
    if not path.is_absolute():
        path = script_dir / path
    return path


def _print_summary(outcomes: list[CellOutcome]) -> None:
    print()
    print("=" * 100)
    print("PHASE B SERIAL RUNNER SUMMARY")
    print("=" * 100)
    by_status: dict[str, int] = {"OK": 0, "SKIP": 0, "FAIL": 0}
    total_runtime = 0.0
    for outcome in outcomes:
        by_status[outcome.status] = by_status.get(outcome.status, 0) + 1
        total_runtime += outcome.elapsed_seconds
        print(
            f"  [{outcome.status:4}] {outcome.dataset:24} "
            f"{outcome.model_a.replace('VLLM/', ''):45} "
            f"{_format_duration(outcome.elapsed_seconds):>7}  {outcome.note}"
        )
    print("-" * 100)
    print(
        f"  total cells={len(outcomes)}  OK={by_status.get('OK', 0)}  "
        f"SKIP={by_status.get('SKIP', 0)}  FAIL={by_status.get('FAIL', 0)}  "
        f"wallclock={_format_duration(total_runtime)}"
    )


def _load_cells(script_dir: Path) -> list[dict[str, object]]:
    cells_path = _cells_path(script_dir)
    if not cells_path.exists():
        raise SystemExit(
            f"{cells_path.name} not found at {cells_path}; the launcher "
            "should have written it next to this runner before submitting."
        )
    with cells_path.open() as fh:
        cells = json.load(fh)
    if not isinstance(cells, list) or not cells:
        raise SystemExit(f"{cells_path.name} at {cells_path} is empty or malformed.")
    return cells


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    cells = _load_cells(script_dir)
    cells_path = _cells_path(script_dir)
    cwd = Path.cwd()
    result_folder = cwd / "results"
    result_folder.mkdir(parents=True, exist_ok=True)

    print("PHASE B SERIAL RUNNER (Gemma-4-31B-IT judge)", flush=True)
    print(f"  cwd={cwd}", flush=True)
    print(f"  cells_path={cells_path}", flush=True)
    print(f"  result_folder={result_folder}", flush=True)
    print(f"  total cells: {len(cells)}", flush=True)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "[fail] OPENROUTER_API_KEY missing from environment; aborting "
            "before any cell runs.",
            flush=True,
        )
        return 2
    if os.environ.get("JUDGEARENA_IGNORE_CACHE", "1") == "1":
        print(
            "[fail] JUDGEARENA_IGNORE_CACHE=1; Phase B requires cache reuse.",
            flush=True,
        )
        return 2
    if os.environ.get("JUDGEARENA_SKIP_JUDGING", "0") == "1":
        print(
            "[fail] JUDGEARENA_SKIP_JUDGING=1; Phase B must run the judge.", flush=True
        )
        return 2

    outcomes: list[CellOutcome] = []
    for idx, cell in enumerate(cells, start=1):
        dataset = str(cell["dataset"])
        model_a = str(cell["model_A"])
        judge_model = str(cell["judge_model"])
        swap_mode = str(cell["swap_mode"])
        args = cell  # build_eval_args output, already includes everything needed
        print()
        print(
            f"[cell {idx}/{len(cells)}] dataset={dataset} model={model_a}",
            flush=True,
        )
        outcome = _run_cell(
            dataset=dataset,
            model_a=model_a,
            judge_model=judge_model,
            swap_mode=swap_mode,
            args=args,
            result_folder=result_folder,
        )
        print(
            f"  -> {outcome.status} ({_format_duration(outcome.elapsed_seconds)})"
            f"{' ' + outcome.note if outcome.note else ''}",
            flush=True,
        )
        outcomes.append(outcome)

    _print_summary(outcomes)
    return 1 if any(o.status == "FAIL" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
