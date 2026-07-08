"""Migrate olmo3-evals CSV/CSV.zip files to the SQLite store layout.

Usage:
    uv run python scripts/migrate_olmo3_to_store.py \
        --data-dir slurmpilot_scripts/olmo3-evals/data \
        --output-dir /path/to/store \
        --judge-model "Together/meta-llama/Llama-3.3-70B-Instruct-Turbo" \
        --pushed-by migration

Output layout:
    {output_dir}/
        completions/{task}/{model_name}/{provider}/
            completions.db
            metadata.json
        judgements/{task}/{judge_name}/{provider}/
            judgements.db
            metadata.json
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from judgearena.store_sqlite import SQLiteCompletionStore, SQLiteJudgementStore

# ---------------------------------------------------------------------------
# Known hash → model mapping (from show_results.py multilingual runs)
# ---------------------------------------------------------------------------

_MULTILINGUAL_LANGS = sorted(
    [
        "hr",
        "cs",
        "en",
        "ro",
        "it",
        "th",
        "eo",
        "fi",
        "hu",
        "uk",
        "ko",
        "el",
        "fa",
        "ja",
        "sv",
        "nl",
        "he",
        "zh",
        "bn",
        "tr",
        "id",
        "sl",
        "de",
        "es",
        "ca",
        "pt",
        "ru",
        "vi",
        "no",
        "fr",
        "pl",
        "sr",
        "ar",
    ]
)
_KNOWN_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.5-9B",
    "allenai/Olmo-3-1025-7B",
    "allenai/Olmo-3-7B-Instruct",
    "allenai/Olmo-3-7B-Think",
    "openeurollm/datamix-9b-Dolci-Translated-A-75EN",
    "swiss-ai/Apertus-8B-Instruct-2509",
    "utter-project/EuroLLM-9B-Instruct-2512",
]


def _build_hash_map() -> dict[str, dict]:
    result = {}
    langs_str = "-".join(_MULTILINGUAL_LANGS)
    for model in _KNOWN_MODELS:
        slug = f"VLLM_{model.replace('/', '_')}"
        suffix = f"LMArena+ComparIA_{slug}_None_100_{langs_str}_8192_32768"
        h = hashlib.sha256(suffix.encode()).hexdigest()[:16]
        result[h] = {
            "model": f"VLLM/{model}",
            "arena": "LMArena+ComparIA",
            "n_instructions": None,
            "n_instructions_per_language": 100,
            "languages": langs_str,
            "truncate_input_chars": 8192,
            "max_out_tokens": 32768,
        }
    return result


_HASH_TO_META = _build_hash_map()
_KNOWN_PROVIDERS = ("VLLM", "OpenRouter", "LlamaCpp")


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def _is_hex16(s: str) -> bool:
    return len(s) == 16 and bool(re.fullmatch(r"[0-9a-f]+", s))


def _parse_filename(stem: str) -> dict | None:
    """Parse a structured filename into metadata. Returns None if unparseable."""
    for provider in _KNOWN_PROVIDERS:
        marker = f"_{provider}_"
        idx = stem.find(marker)
        if idx == -1:
            continue
        arena = stem[:idx]
        rest = stem[idx + len(marker) :]
        parts = rest.split("_")
        if len(parts) < 5:
            return None
        try:
            max_tokens = int(parts[-1])
            truncate = int(parts[-2])
            languages = parts[-3]
            n_per_lang = None if parts[-4] == "None" else int(parts[-4])
            n_instructions = None if parts[-5] == "None" else int(parts[-5])
        except ValueError:
            return None
        model_slug = "_".join(parts[:-5])
        if not model_slug:
            return None
        # Reconstruct HF model path: underscores back to slashes (best effort)
        model_name = model_slug.replace("_", "/")
        return {
            "model": f"{provider}/{model_name}",
            "arena": arena,
            "n_instructions": n_instructions,
            "n_instructions_per_language": n_per_lang,
            "languages": languages,
            "truncate_input_chars": truncate,
            "max_out_tokens": max_tokens,
        }
    return None


def _resolve_meta(real_stem: str) -> dict | None:
    """Return metadata dict for a file stem (hash or structured name)."""
    if _is_hex16(real_stem):
        return _HASH_TO_META.get(real_stem)
    return _parse_filename(real_stem)


def _model_folder(output_dir: Path, meta: dict, kind: str) -> Path:
    """Return the local folder path for a given model/judge config."""
    provider, model_path = meta["model"].split("/", 1)
    model_name = model_path.replace("/", "--")
    arena = meta["arena"]
    return output_dir / kind / arena / model_name / provider


def _judge_folder(output_dir: Path, judge_model: str, arena: str) -> Path:
    provider, model_path = judge_model.split("/", 1)
    judge_name = model_path.replace("/", "--")
    return output_dir / "judgements" / arena / judge_name / provider


def _write_metadata(folder: Path, payload: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def migrate(data_dir: Path, output_dir: Path, judge_model: str, pushed_by: str) -> None:
    now = datetime.now(UTC).isoformat()
    skipped, migrated_comp, migrated_judge = [], 0, 0

    def _stem(p: Path) -> str:
        name = p.name
        for ext in (".csv.zip", ".csv"):
            if name.endswith(ext):
                return name[: -len(ext)]
        return p.stem

    all_files = sorted(data_dir.glob("*.csv*"))
    comp_files = [f for f in all_files if not f.name.startswith("judge_")]
    judge_files = [f for f in all_files if f.name.startswith("judge_")]

    # -- completions ----------------------------------------------------------
    for path in comp_files:
        real_stem = _stem(path)
        meta = _resolve_meta(real_stem)
        if meta is None:
            print(f"  [SKIP] {path.name} — could not parse filename")
            skipped.append(path.name)
            continue

        folder = _model_folder(output_dir, meta, kind="completions")
        _write_metadata(folder, {**meta, "pushed_by": pushed_by, "pushed_at": now})

        df = pd.read_csv(path)
        store = SQLiteCompletionStore(folder / "completions.db")
        n = store.save(df, pushed_by=pushed_by, run_id=real_stem)
        store.close()
        migrated_comp += n
        print(f"  [OK]   {path.name} → {n} completions  ({meta['model']})")

    # -- judgements -----------------------------------------------------------
    for path in judge_files:
        real_stem = _stem(path).removeprefix("judge_")
        meta = _resolve_meta(real_stem)
        if meta is None:
            print(f"  [SKIP] {path.name} — could not parse filename")
            skipped.append(path.name)
            continue

        our_model = meta["model"]
        arena = meta["arena"]
        folder = _judge_folder(output_dir, judge_model, arena)
        _write_metadata(
            folder,
            {
                "model": judge_model,
                "evaluated_model": our_model,
                "pushed_by": pushed_by,
                "pushed_at": now,
            },
        )

        df = pd.read_csv(path)
        df["instruction_index"] = df.index

        # Derive model_A / model_B from position flag
        df["model_A"] = df.apply(
            lambda r, our_model=our_model: (
                our_model if r["our_model_is_position_a"] else r["opponent_model"]
            ),
            axis=1,
        )
        df["model_B"] = df.apply(
            lambda r, our_model=our_model: (
                r["opponent_model"] if r["our_model_is_position_a"] else our_model
            ),
            axis=1,
        )
        df = df.rename(columns={"judge_completion": "judge_output"})

        store = SQLiteJudgementStore(folder / "judgements.db")
        n = store.save(df, pushed_by=pushed_by, run_id=real_stem)
        store.close()
        migrated_judge += n
        print(f"  [OK]   {path.name} → {n} judgements  ({our_model})")

    print(f"\nDone. {migrated_comp} completions, {migrated_judge} judgements migrated.")
    if skipped:
        print(f"Skipped ({len(skipped)}): {skipped}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--judge-model",
        default="Together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        help="Judge model spec used for all judgements in this dataset.",
    )
    parser.add_argument("--pushed-by", default="migration")
    args = parser.parse_args()
    migrate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        judge_model=args.judge_model,
        pushed_by=args.pushed_by,
    )


if __name__ == "__main__":
    main()
