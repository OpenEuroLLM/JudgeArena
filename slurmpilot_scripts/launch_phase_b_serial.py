"""Reproducible launcher for the Phase B benchmark matrix (Gemma-4-31B-IT judge).

Defines the 6-model x 8-dataset cell grid declaratively, freezes it to
``scripts/phase_b_cells.json`` at submit time, and submits a single sequential
Slurm job that runs ``scripts/phase_b_serial_runner.py`` over every cell.

The frozen cell file is an *ephemeral build artifact* (git-ignored): re-running
this launcher regenerates it from scratch, so no hand-patched matrix is kept in
the repo. Each cell carries the full flat config (generation caps + judge
config); the runner translates the flat keys to the current dotted
``judgearena.cli`` flags. A from-scratch run therefore generates the battle
completions and then judges them in the same subprocess; with a warm Phase A
cache present it loads those completions and only judges.

The grid and every constant below are traceable to the run archive at
``judgearena-data/slurmpilot-run-archives/gemma4-report-48-runs-2026-04-28/``
(see ``supporting-files/gemma-4-31b-openrouter-judge-benchmark-results-2026-04-26.md``,
sections 1 and 5) and the per-cell ``args-*.json`` in the sibling result dirs.

Usage (from the repo root, inside the project venv)::

    # dry run: write scripts/phase_b_cells.json and print the job plan
    uv run python slurmpilot_scripts/launch_phase_b_serial.py

    # actually submit the Slurm job (calls OpenRouter -> costs money)
    OPENROUTER_API_KEY=... uv run python slurmpilot_scripts/launch_phase_b_serial.py --submit

Environment overrides: ``JUDGEARENA_CLUSTER``, ``JUDGEARENA_PHASE_B_PARTITION``,
``JUDGEARENA_PHASE_B_MAX_RUNTIME_MINUTES``, ``JUDGEARENA_PYTHON_BINARY``,
``HF_HOME``, ``JUDGEARENA_JUDGE_MAX_CONCURRENCY``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from slurmpilot import JobCreationInfo, SlurmPilot, unify

# --- Judge configuration (constant across all 48 cells; see doc section 5) ---
JUDGE_MODEL = "OpenRouter/google/gemma-4-31b-it"
JUDGE_PROMPT_PRESET = "default"
PROVIDE_EXPLANATION = False
SWAP_MODE = "fixed"
BATTLE_THINKING_TOKEN_BUDGET = 32768
STRIP_THINKING_BEFORE_JUDGING = True
TRUNCATE_JUDGE_INPUT_CHARS = 200_000
MAX_OUT_TOKENS_JUDGE = 16_384
MAX_JUDGE_MODEL_LEN = 200_000

# --- Battle models (model_A): 6 local vLLM models, TP=1 on a single L40S ---
BATTLE_MODELS = [
    "VLLM/Qwen/Qwen3.5-9B",
    "VLLM/HuggingFaceTB/SmolLM3-3B",
    "VLLM/swiss-ai/Apertus-8B-Instruct-2509",
    "VLLM/allenai/Olmo-3-7B-Instruct",
    "VLLM/utter-project/EuroLLM-1.7B-Instruct",
    "VLLM/utter-project/EuroLLM-9B-Instruct",
]

# --- Datasets (8) and their pairwise baselines (model_B). ``None`` lets the
# framework resolve the dataset-native baseline (arena-hard families:
# ``gpt-4-0314`` for v0.1, a per-category ``o3-mini`` map for v2.0). ---
DATASET_BASELINES: dict[str, str | None] = {
    "alpaca-eval": "gpt4_1106_preview",
    "arena-hard-v0.1": None,
    "arena-hard-v2.0": None,
    "mt-bench": "gpt-4",
    "m-arena-hard-v2.0-ar": "google/gemini-2.5-flash",
    "m-arena-hard-v2.0-pl": "google/gemini-2.5-flash",
    "m-arena-hard-v2.0-uk": "google/gemini-2.5-flash",
    "m-arena-hard-v2.0-zh": "google/gemini-2.5-flash",
}

# --- Generation caps (Phase A battle models; see doc section 1) ---
ENGINE_KWARGS = {
    "language_model_only": True,
    "gpu_memory_utilization": 0.90,
    "enforce_eager": True,
}
DEFAULT_GEN_CAPS = {
    "max_model_len": 57_344,
    "max_out_tokens_models": 49_152,
    "truncate_all_input_chars": 30_000,
}
# EuroLLM ships a 4096-token positional cap shared by input+output, so it gets a
# tighter, dataset-dependent input truncation: long-prompt datasets squeeze the
# input to leave answer room, short-prompt datasets keep a loose safety net.
EUROLLM_MAX_MODEL_LEN = 4_096
EUROLLM_MAX_OUT_TOKENS = 4_096
EUROLLM_LONG_PROMPT_TRUNCATE = 3_500
EUROLLM_SHORT_PROMPT_TRUNCATE = 8_000
EUROLLM_LONG_PROMPT_TASKS = frozenset(
    {
        "arena-hard-v2.0",
        "mt-bench",
        "m-arena-hard-v2.0-ar",
        "m-arena-hard-v2.0-pl",
        "m-arena-hard-v2.0-uk",
        "m-arena-hard-v2.0-zh",
    }
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
RUNNER_ENTRYPOINT = "phase_b_serial_runner.py"
CELLS_PATH = SCRIPTS_DIR / "phase_b_cells.json"


def _is_eurollm(model: str) -> bool:
    return "EuroLLM" in model


def _generation_caps(model: str, task: str) -> dict[str, int]:
    if not _is_eurollm(model):
        return dict(DEFAULT_GEN_CAPS)
    truncate = (
        EUROLLM_LONG_PROMPT_TRUNCATE
        if task in EUROLLM_LONG_PROMPT_TASKS
        else EUROLLM_SHORT_PROMPT_TRUNCATE
    )
    return {
        "max_model_len": EUROLLM_MAX_MODEL_LEN,
        "max_out_tokens_models": EUROLLM_MAX_OUT_TOKENS,
        "truncate_all_input_chars": truncate,
    }


def build_eval_args() -> list[dict[str, object]]:
    """Materialize the 6x8 Phase B cell matrix as flat-key arg dicts.

    Keys are the pre-refactor flat CLI names; ``phase_b_serial_runner`` maps
    them to the current dotted ``judgearena.cli`` flags. ``model_B`` is omitted
    for arena-hard tasks so the dataset-native baseline is resolved at runtime.
    """
    cells: list[dict[str, object]] = []
    for task, baseline in DATASET_BASELINES.items():
        for model in BATTLE_MODELS:
            cell: dict[str, object] = {
                "dataset": task,
                "model_A": model,
                "judge_model": JUDGE_MODEL,
                "judge_prompt_preset": JUDGE_PROMPT_PRESET,
                "provide_explanation": PROVIDE_EXPLANATION,
                "swap_mode": SWAP_MODE,
                "battle_thinking_token_budget": BATTLE_THINKING_TOKEN_BUDGET,
                "strip_thinking_before_judging": STRIP_THINKING_BEFORE_JUDGING,
                "truncate_judge_input_chars": TRUNCATE_JUDGE_INPUT_CHARS,
                "max_out_tokens_judge": MAX_OUT_TOKENS_JUDGE,
                "max_judge_model_len": MAX_JUDGE_MODEL_LEN,
                "engine_kwargs": ENGINE_KWARGS,
                "ignore_cache": False,
                "use_tqdm": True,
                **_generation_caps(model, task),
            }
            if baseline is not None:
                cell["model_B"] = baseline
            cells.append(cell)
    return cells


def write_cells_file(cells: list[dict[str, object]], path: Path = CELLS_PATH) -> Path:
    """Freeze the cell matrix next to the runner so SlurmPilot ships it."""
    path.write_text(json.dumps(cells, indent=2) + "\n")
    return path


def _build_env() -> dict[str, str]:
    env = {
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": os.environ.get(
            "HF_HOME", "/work/dlclarge1/lushtake-hiwi/.cache/huggingface"
        ),
        # Phase B reuses cached Phase A completions; the runner aborts if "1".
        "JUDGEARENA_IGNORE_CACHE": "0",
        "JUDGEARENA_JUDGE_MAX_CONCURRENCY": os.environ.get(
            "JUDGEARENA_JUDGE_MAX_CONCURRENCY", "32"
        ),
    }
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        env["OPENROUTER_API_KEY"] = api_key
    return env


def _build_job_info() -> JobCreationInfo:
    cluster = os.environ.get("JUDGEARENA_CLUSTER", "kislurm")
    partition = os.environ.get("JUDGEARENA_PHASE_B_PARTITION", "alldlc2_gpu-l40s")
    max_runtime = int(os.environ.get("JUDGEARENA_PHASE_B_MAX_RUNTIME_MINUTES", 24 * 60))
    python_binary = os.environ.get("JUDGEARENA_PYTHON_BINARY", sys.executable)
    return JobCreationInfo(
        cluster=cluster,
        partition=partition,
        jobname=unify("judgearena/phase-b-serial", method="date"),
        entrypoint=RUNNER_ENTRYPOINT,
        src_dir=str(SCRIPTS_DIR),
        python_binary=python_binary,
        n_cpus=4,
        n_gpus=1,
        max_runtime_minutes=max_runtime,
        env=_build_env(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the Slurm job. Without it, only write the cells file and "
        "print the plan (dry run).",
    )
    submit = parser.parse_args().submit

    cells = build_eval_args()
    write_cells_file(cells)
    print(f"Wrote {len(cells)} cells to {CELLS_PATH}")

    job_info = _build_job_info()
    print(
        f"Job plan: cluster={job_info.cluster} partition={job_info.partition} "
        f"n_gpus={job_info.n_gpus} runtime<={job_info.max_runtime_minutes}min "
        f"python={job_info.python_binary}"
    )

    if not submit:
        print("Dry run (pass --submit to schedule). No job submitted.")
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "OPENROUTER_API_KEY is not set; the Gemma-4 judge calls OpenRouter. "
            "Export it before submitting."
        )

    slurm = SlurmPilot(clusters=[job_info.cluster])
    job_id = slurm.schedule_job(job_info)
    print(f"Job {job_id} scheduled on {job_info.cluster} (partition={job_info.partition})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
