"""Build random1k LMArena Elo tables and plots."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

RESULTS_ROOT = Path("/home/lushtake/slurmpilot/jobs")
OUTPUT_DIR = Path("docs/generated/lmarena_elo")
RESULTS_GLOB = "elo-gemma4-random1k-seed*/results/elo-*/results-*.json"

QWEN_MODELS = [
    "VLLM/Qwen/Qwen3.5-0.8B",
    "VLLM/Qwen/Qwen3.5-2B",
    "VLLM/Qwen/Qwen3.5-4B",
    "VLLM/Qwen/Qwen3.5-9B",
    "VLLM/Qwen/Qwen3.5-27B-FP8",
    "VLLM/Qwen/Qwen3.5-35B-A3B-FP8",
]

CASE_STUDY_MODELS = [
    "VLLM/Qwen/Qwen3.5-9B",
    "VLLM/HuggingFaceTB/SmolLM3-3B",
    "VLLM/allenai/Olmo-3-7B-Instruct",
    "VLLM/allenai/Olmo-3-7B-Think",
    "VLLM/CohereLabs/tiny-aya-global",
    "VLLM/swiss-ai/Apertus-8B-Instruct-2509",
    "VLLM/utter-project/EuroLLM-1.7B-Instruct",
    "VLLM/utter-project/EuroLLM-9B-Instruct",
]

RANDOM1K_ELO_MODELS = list(dict.fromkeys(QWEN_MODELS + CASE_STUDY_MODELS))

MODEL_LABELS = {
    "VLLM/Qwen/Qwen3.5-0.8B": "Qwen3.5-0.8B",
    "VLLM/Qwen/Qwen3.5-2B": "Qwen3.5-2B",
    "VLLM/Qwen/Qwen3.5-4B": "Qwen3.5-4B",
    "VLLM/Qwen/Qwen3.5-9B": "Qwen3.5-9B",
    "VLLM/Qwen/Qwen3.5-27B-FP8": "Qwen3.5-27B-FP8",
    "VLLM/Qwen/Qwen3.5-35B-A3B-FP8": "Qwen3.5-35B-A3B-FP8",
    "VLLM/HuggingFaceTB/SmolLM3-3B": "SmolLM3-3B",
    "VLLM/allenai/Olmo-3-7B-Instruct": "Olmo-3-7B",
    "VLLM/allenai/Olmo-3-7B-Think": "Olmo-3-7B-Think",
    "VLLM/CohereLabs/tiny-aya-global": "Tiny Aya",
    "VLLM/swiss-ai/Apertus-8B-Instruct-2509": "Apertus-8B",
    "VLLM/utter-project/EuroLLM-1.7B-Instruct": "EuroLLM-1.7B",
    "VLLM/utter-project/EuroLLM-9B-Instruct": "EuroLLM-9B",
}

TABLE_COLUMNS = [
    "model_A",
    "model_label",
    "arena",
    "elo_mean",
    "elo_std",
    "elo_ci_low",
    "elo_ci_high",
    "elo_num_bootstraps",
    "num_battles",
    "llm_judged_battles",
    "human_anchor_battles",
    "sampling_mode",
    "random_seed",
    "requested_rows",
    "sampled_rows",
    "sample_fingerprint",
    "source_job",
    "source_path",
]

_VIRIDIS = plt.get_cmap("viridis")


def _parse_results(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def _summary(payload: dict) -> dict:
    return payload.get("summary", payload)


def _elo_bootstrap_values(payload: dict, model: str) -> np.ndarray:
    values = [
        float(rating[model])
        for rating in payload.get("bootstrap_ratings", [])
        if model in rating
    ]
    return np.asarray(values, dtype=float)


def _elo_bootstrap_ci(payload: dict, model: str) -> tuple[float, float]:
    values = _elo_bootstrap_values(payload, model)
    if values.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _candidate_row(path: Path) -> dict[str, object] | None:
    payload = _parse_results(path)
    summary = _summary(payload)
    metadata = summary.get("sampling_metadata", {}) or {}
    if metadata.get("sampling_mode") != "seeded_random":
        return None
    if int(metadata.get("requested_rows", -1)) != 1000:
        return None
    if int(metadata.get("random_seed", -1)) != 0:
        return None
    model = str(summary["model_A"])
    if model not in RANDOM1K_ELO_MODELS:
        return None
    elo_ci_low, elo_ci_high = _elo_bootstrap_ci(payload, model)
    return {
        "model_A": model,
        "model_label": MODEL_LABELS[model],
        "arena": summary.get("arena"),
        "elo_mean": summary.get("elo_mean"),
        "elo_std": summary.get("elo_std"),
        "elo_ci_low": elo_ci_low,
        "elo_ci_high": elo_ci_high,
        "elo_num_bootstraps": summary.get("elo_num_bootstraps"),
        "num_battles": summary.get("num_battles"),
        "llm_judged_battles": summary.get("llm_judged_battles"),
        "human_anchor_battles": summary.get("human_anchor_battles"),
        "sampling_mode": metadata.get("sampling_mode"),
        "random_seed": metadata.get("random_seed"),
        "requested_rows": metadata.get("requested_rows"),
        "sampled_rows": metadata.get("sampled_rows"),
        "sample_fingerprint": metadata.get("sample_fingerprint"),
        "source_job": path.parents[2].name,
        "source_path": str(path),
    }


def build_elo_dataframe(
    results_root: Path = RESULTS_ROOT,
    *,
    results_glob: str = RESULTS_GLOB,
    require_models: Iterable[str] | None = None,
) -> pd.DataFrame:
    latest_by_model: dict[str, dict[str, object]] = {}
    for path in sorted(results_root.glob(results_glob)):
        row = _candidate_row(path)
        if row is None:
            continue
        existing = latest_by_model.get(str(row["model_A"]))
        if existing is None or str(row["source_path"]) > str(existing["source_path"]):
            latest_by_model[str(row["model_A"])] = row

    required = list(require_models or [])
    if required:
        missing = sorted(set(required) - set(latest_by_model))
        if missing:
            raise FileNotFoundError(
                "Missing random1k Elo results for: " + ", ".join(missing)
            )

    rows = list(latest_by_model.values())
    if not rows:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    fingerprints = {row.get("sample_fingerprint") for row in rows}
    if len(fingerprints) != 1:
        raise ValueError(f"Expected one sample_fingerprint, got {sorted(fingerprints)}")

    df = pd.DataFrame(rows)
    df["model_A"] = pd.Categorical(
        df["model_A"], categories=RANDOM1K_ELO_MODELS, ordered=True
    )
    df = df.sort_values("model_A").reset_index(drop=True)
    df["model_A"] = df["model_A"].astype(str)
    return df.loc[:, TABLE_COLUMNS]


def _save_figure(fig: plt.Figure, pdf_path: Path, png_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _colors(models: Sequence[str]) -> list[str]:
    values = np.linspace(0.08, 0.92, len(models))
    return [to_hex(_VIRIDIS(value)) for value in values]


def _write_barplot(
    df: pd.DataFrame,
    *,
    models: Sequence[str],
    title: str,
    pdf_path: Path,
    png_path: Path,
) -> None:
    ordered = df.set_index("model_A").reindex(models)
    values = ordered["elo_mean"].to_numpy(dtype=float)
    ci_low = ordered["elo_ci_low"].to_numpy(dtype=float)
    ci_high = ordered["elo_ci_high"].to_numpy(dtype=float)
    errors = np.vstack(
        [
            np.maximum(values - ci_low, 0.0),
            np.maximum(ci_high - values, 0.0),
        ]
    )
    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.bar(
        x,
        values,
        width=0.55,
        color=_colors(models),
        yerr=errors,
        capsize=2,
        error_kw={"linewidth": 0.8},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[model] for model in models], rotation=25, ha="right"
    )
    ax.set_ylabel("Estimated Elo")
    ax.set_xlabel("")
    ax.set_ylim(0, float(np.nanmax(values + errors) * 1.08))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    _save_figure(fig, pdf_path, png_path)


def _write_case_study_table(df: pd.DataFrame, path: Path) -> None:
    ordered = df.set_index("model_A").reindex(CASE_STUDY_MODELS)
    ordered = ordered.sort_values("elo_mean", ascending=False)
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Model} & \textbf{LMArena Elo} & \textbf{95\% CI} \\",
        r"\midrule",
    ]
    for model, row in ordered.iterrows():
        lines.append(
            f"{MODEL_LABELS[str(model)]} & {float(row['elo_mean']):.1f} & "
            f"[{float(row['elo_ci_low']):.1f}, {float(row['elo_ci_high']):.1f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))


def _subset(df: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    return df[df["model_A"].isin(models)].copy()


def write_outputs(df: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    qwen = _subset(df, QWEN_MODELS)
    case_study = _subset(df, CASE_STUDY_MODELS)

    qwen_csv = output_dir / "qwen_lmarena_elo_random1k_seed0_results.csv"
    qwen_json = output_dir / "qwen_lmarena_elo_random1k_seed0_results.json"
    qwen_pdf = output_dir / "qwen_lmarena_elo_random1k_seed0.pdf"
    qwen_png = output_dir / "qwen_lmarena_elo_random1k_seed0.png"
    case_csv = output_dir / "case_study_lmarena_elo_random1k_seed0_results.csv"
    case_json = output_dir / "case_study_lmarena_elo_random1k_seed0_results.json"
    case_pdf = output_dir / "case_study_lmarena_elo_random1k_seed0.pdf"
    case_png = output_dir / "case_study_lmarena_elo_random1k_seed0.png"
    case_tex = output_dir / "case_study_lmarena_elo_random1k_seed0_table.tex"

    qwen.to_csv(qwen_csv, index=False)
    qwen_json.write_text(qwen.to_json(orient="records", indent=2) + "\n")
    case_study.to_csv(case_csv, index=False)
    case_json.write_text(case_study.to_json(orient="records", indent=2) + "\n")

    _write_barplot(
        qwen,
        models=QWEN_MODELS,
        title="Qwen Estimated LMArena-140K Elo (Random 1K, Seed 0)",
        pdf_path=qwen_pdf,
        png_path=qwen_png,
    )
    _write_barplot(
        case_study,
        models=CASE_STUDY_MODELS,
        title="Case Study Estimated LMArena-140K Elo (Random 1K, Seed 0)",
        pdf_path=case_pdf,
        png_path=case_png,
    )
    _write_case_study_table(case_study, case_tex)

    return {
        "qwen_csv": qwen_csv,
        "qwen_json": qwen_json,
        "qwen_plot_pdf": qwen_pdf,
        "qwen_plot_png": qwen_png,
        "case_study_csv": case_csv,
        "case_study_json": case_json,
        "case_study_plot_pdf": case_pdf,
        "case_study_plot_png": case_png,
        "case_study_table_tex": case_tex,
    }


def run_analysis(
    results_root: Path = RESULTS_ROOT,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    df = build_elo_dataframe(results_root, require_models=RANDOM1K_ELO_MODELS)
    written = write_outputs(df, output_dir)
    return df, written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    df, written = run_analysis(args.results_root, args.output_dir)
    print(f"Rows: {len(df)}")
    print(f"Sample fingerprint: {df['sample_fingerprint'].iloc[0]}")
    print("Wrote artifacts:")
    for key, path in written.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
