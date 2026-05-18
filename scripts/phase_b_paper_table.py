"""Build the Phase B leaderboard LaTeX table for the paper.

Reads all 64 Phase B `results-*.json` files, computes per-cell winrates with
95 % percentile bootstrap CIs on `preferences`, and emits a LaTeX `tabular`
fragment ready to drop into `judge_arena_neurips2026.tex` as
`tab:benchmark_results`.

The discrete winrate convention matches `judgearena.utils.compute_pref_summary`:
`prefs < 0.5` is a win for model_A, `prefs > 0.5` a loss, `prefs == 0.5` a
tie, and `winrate = (num_wins + 0.5 * num_ties) / num_battles`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_RESULTS_ROOT = Path("/home/lushtake/slurmpilot/jobs")
DEFAULT_RESULTS_GLOB = "phase-b-serial-*/results/*/results-*.json"
DEFAULT_ELO_CSV = Path(
    "docs/generated/lmarena_elo/case_study_lmarena_elo_random1k_seed0_results.csv"
)

DATASET_ORDER: list[str] = [
    "alpaca-eval",
    "arena-hard-v0.1",
    "arena-hard-v2.0",
    "mt-bench",
    "m-arena-hard-v2.0-ar",
    "m-arena-hard-v2.0-pl",
    "m-arena-hard-v2.0-uk",
    "m-arena-hard-v2.0-zh",
]

DATASET_HEADER: dict[str, str] = {
    "alpaca-eval": "AlpacaEval",
    "arena-hard-v0.1": "AH v0.1",
    "arena-hard-v2.0": "AH v2.0",
    "mt-bench": "MT-Bench",
    "m-arena-hard-v2.0-ar": "AR",
    "m-arena-hard-v2.0-pl": "PL",
    "m-arena-hard-v2.0-uk": "UK",
    "m-arena-hard-v2.0-zh": "ZH",
}

MODEL_HEADER: dict[str, str] = {
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "allenai/Olmo-3-7B-Instruct": "Olmo-3-7B",
    "allenai/Olmo-3-7B-Think": "Olmo-3-7B-Think",
    "CohereLabs/tiny-aya-global": "Tiny Aya",
    "HuggingFaceTB/SmolLM3-3B": "SmolLM3-3B",
    "swiss-ai/Apertus-8B-Instruct-2509": "Apertus-8B",
    "utter-project/EuroLLM-9B-Instruct": "EuroLLM-9B",
    "utter-project/EuroLLM-1.7B-Instruct": "EuroLLM-1.7B",
}

EXPECTED_CELL_COUNT = len(DATASET_ORDER) * len(MODEL_HEADER)
TABLE_TRAILING_HEADERS = [r"\textbf{Overall}", r"\textbf{LMArena Elo}"]

ARCH_CONSTRAINED_LONG_PROMPT_DATASETS: set[str] = {
    "arena-hard-v2.0",
    "m-arena-hard-v2.0-ar",
    "m-arena-hard-v2.0-pl",
    "m-arena-hard-v2.0-uk",
    "m-arena-hard-v2.0-zh",
    "mt-bench",
}
ARCH_CONSTRAINED_MODELS: set[str] = {
    "utter-project/EuroLLM-1.7B-Instruct",
    "utter-project/EuroLLM-9B-Instruct",
}


def discrete_winrate(prefs: np.ndarray) -> float:
    valid = prefs[~np.isnan(prefs)]
    if valid.size == 0:
        return float("nan")
    wins = int((valid < 0.5).sum())
    ties = int((valid == 0.5).sum())
    return (wins + 0.5 * ties) / float(valid.size)


def bootstrap_winrate_ci(
    prefs: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    valid = prefs[~np.isnan(prefs)]
    if valid.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = valid.size
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = valid[rng.integers(0, n, size=n)]
        wins = (sample < 0.5).sum()
        ties = (sample == 0.5).sum()
        boot[i] = (wins + 0.5 * ties) / float(n)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def load_cell(path: Path) -> tuple[str, str, np.ndarray, dict]:
    with open(path) as f:
        d = json.load(f)
    summary = d.get("summary") or d
    dataset = summary.get("dataset") or summary.get("task")
    if dataset is None:
        raise KeyError("dataset")
    model_a = summary["model_A"]
    if model_a.startswith("VLLM/"):
        model_a = model_a[len("VLLM/") :]
    prefs = np.asarray(
        d.get("preferences") or summary.get("preferences") or [], dtype=float
    )
    return dataset, model_a, prefs, d


def find_cells(root: Path, results_glob: str = DEFAULT_RESULTS_GLOB) -> list[Path]:
    if root.name == "results":
        candidates = sorted(root.glob("*/results-*.json"))
    else:
        candidates = sorted(root.glob(results_glob))

    latest: dict[tuple[str, str], Path] = {}
    for path in candidates:
        dataset, model, _prefs, _raw = load_cell(path)
        if dataset not in DATASET_ORDER or model not in MODEL_HEADER:
            continue
        key = (dataset, model)
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path
    return [latest[key] for key in sorted(latest)]


def fmt_winrate(wr: float, low: float, high: float, arch: bool) -> str:
    dagger = r"{}^{\dagger}" if arch else ""
    return f"${wr:.2f}_{{{low:.2f}}}^{{{high:.2f}}}{dagger}$"


def fmt_overall(wr: float) -> str:
    return f"\\textbf{{{wr:.3f}}}"


def fmt_elo(mean: float, low: float, high: float) -> str:
    return f"${mean:.0f}_{{{low:.0f}}}^{{{high:.0f}}}$"


def overall_benchmark_average(
    table: dict[tuple[str, str], dict],
    model: str,
    datasets: list[str],
) -> float:
    values = [float(table[(dataset, model)]["winrate"]) for dataset in datasets]
    return float(np.mean(values))


def table_preamble_lines() -> list[str]:
    return [
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l rrrr rrrr r r}",
    ]


def table_header_line(benchmark_headers: list[str], trailing_headers: list[str]) -> str:
    header_cells = [r"\textbf{Model}"] + benchmark_headers + trailing_headers
    return " & ".join(header_cells) + r" \\"


def _strip_vllm_prefix(model: str) -> str:
    return model[len("VLLM/") :] if model.startswith("VLLM/") else model


def load_lmarena_elo(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"missing LMArena Elo CSV: {path}")

    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    elo: dict[str, dict[str, float]] = {}
    for row in rows:
        model = _strip_vllm_prefix(row["model_A"])
        if model not in MODEL_HEADER:
            continue
        elo[model] = {
            "mean": float(row["elo_mean"]),
            "low": float(row["elo_ci_low"]),
            "high": float(row["elo_ci_high"]),
        }

    missing = sorted(set(MODEL_HEADER) - set(elo))
    if missing:
        raise FileNotFoundError("missing LMArena Elo rows for: " + ", ".join(missing))
    return elo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Phase B Slurmpilot jobs root or one job's results directory.",
    )
    parser.add_argument(
        "--results-glob",
        default=DEFAULT_RESULTS_GLOB,
        help="Glob used when --root points at the Slurmpilot jobs root.",
    )
    parser.add_argument(
        "--elo-csv",
        type=Path,
        default=DEFAULT_ELO_CSV,
        help="Random1k LMArena Elo CSV with quantile CI columns.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples per cell.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the LaTeX fragment to. Always also prints to stdout.",
    )
    args = parser.parse_args()

    cells = find_cells(args.root, args.results_glob)
    if len(cells) != EXPECTED_CELL_COUNT:
        raise SystemExit(
            f"expected {EXPECTED_CELL_COUNT} cells, found {len(cells)} under {args.root}"
        )
    lmarena_elo = load_lmarena_elo(args.elo_csv)

    table: dict[tuple[str, str], dict] = {}
    for path in cells:
        dataset, model, prefs, _raw = load_cell(path)
        wr = discrete_winrate(prefs)
        arch = (
            model in ARCH_CONSTRAINED_MODELS
            and dataset in ARCH_CONSTRAINED_LONG_PROMPT_DATASETS
        )
        ci_low, ci_high = bootstrap_winrate_ci(
            prefs,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        table[(dataset, model)] = {
            "winrate": wr,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": int(prefs.size),
            "arch": arch,
        }

    overall: dict[str, dict] = {}
    for model in MODEL_HEADER:
        wr = overall_benchmark_average(table, model, DATASET_ORDER)
        overall[model] = {
            "winrate": wr,
        }

    model_order = sorted(MODEL_HEADER.keys(), key=lambda m: -overall[m]["winrate"])

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        r"\caption{Win-rate of eight open models against each benchmark's "
        r"default reference baseline, judged by Gemma-4-31B-IT (OpenRouter) under "
        r"the tuned \texttt{default} preset (one judge call per battle, "
        r"\texttt{swap\_mode=fixed}). Each cell averages "
        r"$\mathbf{1}[\sigma(\Delta\text{score}) < 0.5] + 0.5 \cdot \mathbf{1}[\cdot = 0.5]$ "
        r"over the benchmark's prompts (per-cell $n$ matches its native size: "
        r"AlpacaEval 805, AH v0.1 500, AH v2.0 750, MT-Bench 160, m-AH v2.0 "
        r"498). Benchmark cells report empirical 95\% CIs from 1\,000 "
        r"bootstrap resamples as subscripts/superscripts. The \textit{Overall} "
        r"column averages the eight benchmark-level win-rate point estimates "
        r"and reports that point estimate only. The LMArena Elo column uses the seeded random 1K "
        r"English battle setup with the same CI convention. "
        r"Cells marked with $^{\dagger}$ are subject "
        r"to EuroLLM's 4096-token positional cap on long-prompt benchmarks "
        r"(\cref{sec:exp_cross_benchmark}); the underlying input is truncated for "
        r"$\sim$2--9\,\% of rows.}"
    )
    lines.append(r"\label{tab:benchmark_results}")
    lines.extend(table_preamble_lines())
    lines.append(r"\toprule")
    lines.append(
        r" & \multicolumn{4}{c}{\textbf{English / single-language}} & "
        r"\multicolumn{4}{c}{\textbf{m-Arena-Hard v2.0}} & & \\"
    )
    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(
        table_header_line(
            [DATASET_HEADER[d] for d in DATASET_ORDER],
            TABLE_TRAILING_HEADERS,
        )
    )
    lines.append(r"\midrule")

    for model in model_order:
        row = [MODEL_HEADER[model]]
        for dataset in DATASET_ORDER:
            cell = table[(dataset, model)]
            row.append(
                fmt_winrate(
                    cell["winrate"],
                    cell["ci_low"],
                    cell["ci_high"],
                    cell["arch"],
                )
            )
        ov = overall[model]
        row.append(fmt_overall(ov["winrate"]))
        elo = lmarena_elo[model]
        row.append(fmt_elo(elo["mean"], elo["low"], elo["high"]))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    out = "\n".join(lines) + "\n"
    print(out)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out)
        print(f"\nwrote LaTeX fragment to {args.out}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
