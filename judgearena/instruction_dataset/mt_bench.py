import warnings
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.dataset_revisions import RAW_URL_REVISIONS, hf_revision
from judgearena.utils import data_root


def _fastchat_reference_url() -> str:
    """URL for FastChat MT-Bench GPT-4 references, pinned when available."""
    revision = RAW_URL_REVISIONS.get("lm-sys/FastChat")
    rev = revision if revision else "main"
    return (
        f"https://raw.githubusercontent.com/lm-sys/FastChat/{rev}/"
        "fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl"
    )


FASTCHAT_GPT4_REFERENCE_URL = _fastchat_reference_url()


def _download_gpt4_references(local_dir: Path) -> Path | None:
    reference_dir = local_dir / "reference_answer"
    reference_dir.mkdir(parents=True, exist_ok=True)
    gpt4_reference_path = reference_dir / "gpt-4.jsonl"
    if gpt4_reference_path.exists():
        return gpt4_reference_path
    try:
        urlretrieve(FASTCHAT_GPT4_REFERENCE_URL, gpt4_reference_path)
    except Exception as e:
        warnings.warn(
            "Could not download MT-Bench GPT-4 reference answers from FastChat. "
            f"Falling back to inline references from question.jsonl: {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return gpt4_reference_path


def download_mt_bench(local_dir: Path | None = None) -> tuple[Path, Path | None]:
    """Download MT-Bench questions and GPT-4 references if missing."""
    if local_dir is None:
        local_dir = data_root / "mt-bench"
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"Cannot create MT-Bench cache directory at {local_dir}. "
            "Set environment variable OPENJURY_DATA to a writable location."
        ) from e

    question_path = local_dir / "data" / "mt_bench" / "question.jsonl"
    if not question_path.exists():
        try:
            mt_bench_repo = "lmsys/mt-bench"
            snapshot_download(
                repo_id=mt_bench_repo,
                repo_type="space",
                allow_patterns=[
                    "data/mt_bench/question.jsonl",
                ],
                local_dir=local_dir,
                force_download=False,
                revision=hf_revision(mt_bench_repo),
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to download MT-Bench questions from HuggingFace space "
                "'lmsys/mt-bench'. If you're in an offline / restricted-network "
                "environment, pre-download the space snapshot and place the "
                f"questions file at {question_path}, or set OPENJURY_DATA to "
                "point to that directory."
            ) from e
    if not question_path.exists():
        raise FileNotFoundError(
            "Could not locate MT-Bench questions after download. "
            f"Expected file at {question_path}."
        )

    gpt4_reference_path = _download_gpt4_references(local_dir)
    return question_path, gpt4_reference_path


def load_mt_bench() -> pd.DataFrame:
    """Load MT-Bench questions and reference answers.

    Downloads MT-Bench questions from the HuggingFace LMSYS space and tries to
    load GPT-4 references from FastChat GitHub. If GPT-4 references cannot be
    downloaded or parsed, falls back to inline references from question.jsonl.
    """
    question_path, ref_path = download_mt_bench()

    questions = pd.read_json(question_path, lines=True).to_dict(orient="records")

    ref_by_id: dict[int | str, list[str]] = {}
    use_inline_reference_fallback = ref_path is None
    if ref_path is not None:
        try:
            reference_records = pd.read_json(ref_path, lines=True).to_dict(
                orient="records"
            )
            for rec in reference_records:
                qid = rec.get("question_id", rec.get("id"))
                if qid is None:
                    continue
                choices = rec.get("choices")
                if not (isinstance(choices, list) and choices):
                    continue
                first_choice = choices[0]
                if not isinstance(first_choice, dict):
                    continue
                turns = first_choice.get("turns")
                if not isinstance(turns, list):
                    continue
                ref_by_id[qid] = turns
                try:
                    ref_by_id[int(qid)] = turns
                except Exception:
                    pass
        except Exception as e:
            warnings.warn(
                "Failed to parse GPT-4 reference answers from FastChat. "
                f"Falling back to inline references from question.jsonl: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            use_inline_reference_fallback = True

    rows = []
    for rec in questions:
        qid_raw = rec.get("question_id", rec.get("id"))
        if qid_raw is None:
            raise ValueError(
                f"MT-Bench question record missing question_id/id: keys={list(rec.keys())}"
            )
        try:
            qid = int(qid_raw)
        except Exception:
            qid = qid_raw

        category = rec.get("category")
        turns = rec.get("turns")
        if isinstance(turns, list):
            turn_1 = turns[0] if len(turns) > 0 else None
            turn_2 = turns[1] if len(turns) > 1 else None
        else:
            turn_1 = rec.get("turn_1", rec.get("instruction"))
            turn_2 = rec.get("turn_2")

        ref_turns = ref_by_id.get(qid_raw) or ref_by_id.get(qid)
        if ref_turns is None and use_inline_reference_fallback:
            inline_ref = rec.get("reference")
            if isinstance(inline_ref, list):
                ref_turns = inline_ref

        ref_turn_1 = (
            ref_turns[0] if isinstance(ref_turns, list) and len(ref_turns) > 0 else None
        )
        ref_turn_2 = (
            ref_turns[1] if isinstance(ref_turns, list) and len(ref_turns) > 1 else None
        )

        rows.append(
            {
                "instruction_index": qid,
                "category": category,
                "turn_1": turn_1,
                "turn_2": turn_2,
                "reference_turn_1": ref_turn_1,
                "reference_turn_2": ref_turn_2,
                "instruction": turn_1,
            }
        )

    return pd.DataFrame(rows)
