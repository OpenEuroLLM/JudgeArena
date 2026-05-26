"""Build the localized m-Arena-Hard judge-prompt ablation table.

The table uses only the Phase B runs with m-Arena-Hard v2.0 language-localized
judge prompts. It mirrors the case-study leaderboard style, but restricts the
columns to AR/PL/UK/ZH and reports the average rank across those localized
benchmark cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_RESULTS_ROOT = Path("/home/lushtake/slurmpilot/jobs")
DEFAULT_RESULTS_GLOB = (
    "phase-b-serial-marena-localized-prompts-*/results/*/results-*.json"
)

DATASET_ORDER: list[str] = [
    "m-arena-hard-v2.0-ar",
    "m-arena-hard-v2.0-pl",
    "m-arena-hard-v2.0-uk",
    "m-arena-hard-v2.0-zh",
]

DATASET_HEADER: dict[str, str] = {
    "m-arena-hard-v2.0-ar": "AR",
    "m-arena-hard-v2.0-pl": "PL",
    "m-arena-hard-v2.0-uk": "UK",
    "m-arena-hard-v2.0-zh": "ZH",
}

DATASET_PROMPT_PRESET: dict[str, str] = {
    "m-arena-hard-v2.0-ar": "m-arena-hard-v2-localized-ar",
    "m-arena-hard-v2.0-pl": "m-arena-hard-v2-localized-pl",
    "m-arena-hard-v2.0-uk": "m-arena-hard-v2-localized-uk",
    "m-arena-hard-v2.0-zh": "m-arena-hard-v2-localized-zh",
}

MODEL_HEADER: dict[str, str] = {
    "Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "allenai/Olmo-3-7B-Think": "Olmo-3-7B-Think",
    "allenai/Olmo-3-7B-Instruct": "Olmo-3-7B",
    "HuggingFaceTB/SmolLM3-3B": "SmolLM3-3B",
    "CohereLabs/tiny-aya-global": "Tiny Aya",
    "swiss-ai/Apertus-8B-Instruct-2509": "Apertus-8B",
    "utter-project/EuroLLM-9B-Instruct": "EuroLLM-9B",
    "utter-project/EuroLLM-1.7B-Instruct": "EuroLLM-1.7B",
}

EXPECTED_CELL_COUNT = len(DATASET_ORDER) * len(MODEL_HEADER)

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


def _strip_vllm_prefix(model: str) -> str:
    return model[len("VLLM/") :] if model.startswith("VLLM/") else model


def load_cell(path: Path) -> tuple[str, str, str, np.ndarray, dict]:
    with path.open() as fh:
        data = json.load(fh)

    summary = data.get("summary") or data
    dataset = summary.get("dataset") or summary.get("task")
    if dataset is None:
        raise KeyError("dataset")
    model_a = _strip_vllm_prefix(summary["model_A"])
    prompt_preset = summary.get("judge_prompt_preset")
    prefs = np.asarray(data.get("preferences") or summary.get("preferences") or [])
    return dataset, model_a, prompt_preset, prefs.astype(float), data


def _is_expected_localized_cell(dataset: str, model: str, prompt_preset: str) -> bool:
    return (
        dataset in DATASET_ORDER
        and model in MODEL_HEADER
        and prompt_preset == DATASET_PROMPT_PRESET[dataset]
    )


def find_cells(
    root: Path,
    results_glob: str = DEFAULT_RESULTS_GLOB,
) -> list[Path]:
    if root.name == "results":
        candidates = sorted(root.glob("*/results-*.json"))
    else:
        candidates = sorted(root.glob(results_glob))

    latest: dict[tuple[str, str], Path] = {}
    for path in candidates:
        dataset, model, prompt_preset, _prefs, _raw = load_cell(path)
        if not _is_expected_localized_cell(dataset, model, prompt_preset):
            continue
        key = (dataset, model)
        if key not in latest or path.stat().st_mtime > latest[key].stat().st_mtime:
            latest[key] = path
    return [latest[key] for key in sorted(latest)]


def fmt_winrate(wr: float, low: float, high: float, arch: bool) -> str:
    dagger = r"{}^{\dagger}" if arch else ""
    return f"${wr:.2f}_{{{low:.2f}}}^{{{high:.2f}}}{dagger}$"


def fmt_avg_rank(rank: float) -> str:
    return f"\\textbf{{{rank:.3f}}}"


def localized_average_rank(
    table: dict[tuple[str, str], dict],
    model: str,
    datasets: list[str] = DATASET_ORDER,
    models: list[str] | None = None,
) -> float:
    ranks = _localized_rank_table(table, datasets=datasets, models=models)
    return float(ranks[model])


def _localized_rank_table(
    table: dict[tuple[str, str], dict],
    datasets: list[str] = DATASET_ORDER,
    models: list[str] | None = None,
) -> dict[str, float]:
    models = models or list(MODEL_HEADER)
    rank_sums = {model: 0.0 for model in models}
    for dataset in datasets:
        scores = {model: float(table[(dataset, model)]["winrate"]) for model in models}
        for model, rank in _average_descending_ranks(scores).items():
            rank_sums[model] += rank
    return {model: rank_sums[model] / len(datasets) for model in models}


def _average_descending_ranks(scores: dict[str, float]) -> dict[str, float]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ranks: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and abs(ordered[j][1] - ordered[i][1]) < 1e-12:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for index in range(i, j):
            ranks[ordered[index][0]] = avg_rank
        i = j
    return ranks


def render_table(
    table: dict[tuple[str, str], dict],
    model_order: list[str] | None = None,
) -> str:
    candidate_order = model_order or list(MODEL_HEADER)
    avg_ranks = _localized_rank_table(table, models=candidate_order)
    sorted_models = sorted(candidate_order, key=lambda model: avg_ranks[model])

    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(
        r"\caption{Localized judge-prompt ablation on m-Arena-Hard~v2.0. "
        r"The evaluated model completions and Gemma-4-31B-IT judge route match "
        r"\cref{tab:benchmark_results}, but each language uses a translated "
        r"score-style judge prompt instead of the English \texttt{default} "
        r"prompt. Each cell reports the evaluated model's win-rate against "
        r"Gemini 2.5 Flash with empirical 95\% confidence intervals from "
        r"1\,000 bootstrap resamples; \textit{Avg Rank} reports the mean rank "
        r"over AR/PL/UK/ZH, with lower being better.}"
    )
    lines.append(r"\label{tab:marena_hard_localized_prompt_results}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\resizebox{0.82\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l rrrr r}")
    lines.append(r"\toprule")
    lines.append(
        r" & \multicolumn{4}{c}{\textbf{Localized m-Arena-Hard v2.0 prompts}} & \\"
    )
    lines.append(r"\cmidrule(lr){2-5}")
    header_cells = [r"\textbf{Model}"]
    header_cells.extend(DATASET_HEADER[dataset] for dataset in DATASET_ORDER)
    header_cells.append(r"\textbf{Avg Rank}")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for model in sorted_models:
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
        row.append(fmt_avg_rank(avg_ranks[model]))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def build_table(
    cells: list[Path],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[tuple[str, str], dict]:
    table: dict[tuple[str, str], dict] = {}
    for path in cells:
        dataset, model, _prompt_preset, prefs, _raw = load_cell(path)
        wr = discrete_winrate(prefs)
        ci_low, ci_high = bootstrap_winrate_ci(
            prefs,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        table[(dataset, model)] = {
            "winrate": wr,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": int(prefs.size),
            "arch": model in ARCH_CONSTRAINED_MODELS,
        }
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Slurmpilot jobs root or one job's results directory.",
    )
    parser.add_argument(
        "--results-glob",
        default=DEFAULT_RESULTS_GLOB,
        help="Glob used when --root points at the Slurmpilot jobs root.",
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
        help="Optional path to write the LaTeX table. Always also prints to stdout.",
    )
    args = parser.parse_args()

    cells = find_cells(args.root, args.results_glob)
    if len(cells) != EXPECTED_CELL_COUNT:
        raise SystemExit(
            f"expected {EXPECTED_CELL_COUNT} localized cells, found {len(cells)} "
            f"under {args.root}"
        )

    table = build_table(cells, n_bootstrap=args.n_bootstrap, seed=args.seed)
    rendered = render_table(table)
    print(rendered)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"\nwrote LaTeX table to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
