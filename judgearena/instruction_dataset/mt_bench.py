import warnings
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.utils import data_root

MT_BENCH_SPACE_ID = "lmsys/mt-bench"
MT_BENCH_QUESTION_PATTERN = "data/mt_bench/question.jsonl"
MT_BENCH_MODEL_ANSWER_DIR = Path("data") / "mt_bench" / "model_answer"
FASTCHAT_GPT4_REFERENCE_URL = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/"
    "fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl"
)


def _normalize_question_id(question_id: object) -> object:
    try:
        return int(question_id)
    except Exception:
        return question_id


def _snapshot_mt_bench_files(
    *,
    local_dir: Path,
    allow_patterns: list[str],
    expected_path: Path,
    description: str,
) -> None:
    try:
        snapshot_download(
            repo_id=MT_BENCH_SPACE_ID,
            repo_type="space",
            allow_patterns=allow_patterns,
            local_dir=local_dir,
            force_download=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {description} from HuggingFace space "
            f"'{MT_BENCH_SPACE_ID}'. If you're in an offline / restricted-network "
            f"environment, pre-download the space snapshot and place the file at "
            f"{expected_path}, or set OPENJURY_DATA to point to that directory."
        ) from e
    if not expected_path.exists():
        raise FileNotFoundError(
            f"Could not locate {description} after download. "
            f"Expected file at {expected_path}."
        )


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
        _snapshot_mt_bench_files(
            local_dir=local_dir,
            allow_patterns=[MT_BENCH_QUESTION_PATTERN],
            expected_path=question_path,
            description="MT-Bench questions",
        )

    gpt4_reference_path = _download_gpt4_references(local_dir)
    return question_path, gpt4_reference_path


def download_mt_bench_model_answer(
    model_id: str, local_dir: Path | None = None
) -> Path:
    """Download a cached MT-Bench baseline answer file if missing."""
    if local_dir is None:
        local_dir = data_root / "mt-bench"
    answer_path = local_dir / MT_BENCH_MODEL_ANSWER_DIR / f"{model_id}.jsonl"
    if answer_path.exists():
        return answer_path
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    allow_pattern = (MT_BENCH_MODEL_ANSWER_DIR / f"{model_id}.jsonl").as_posix()
    _snapshot_mt_bench_files(
        local_dir=local_dir,
        allow_patterns=[allow_pattern],
        expected_path=answer_path,
        description=f"MT-Bench model answers for '{model_id}'",
    )
    return answer_path


def _extract_answer_turns(record: dict, source_name: str) -> tuple[object, list[str]]:
    question_id = record.get("question_id", record.get("id"))
    if question_id is None:
        raise ValueError(
            f"MT-Bench answer record from {source_name} is missing question_id/id."
        )
    choices = record.get("choices")
    if not (isinstance(choices, list) and choices):
        raise ValueError(
            f"MT-Bench answer record for question {question_id} in {source_name} is "
            "missing a non-empty choices list."
        )
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError(
            f"MT-Bench answer record for question {question_id} in {source_name} has "
            "a malformed first choice entry."
        )
    turns = first_choice.get("turns")
    if not isinstance(turns, list):
        raise ValueError(
            f"MT-Bench answer record for question {question_id} in {source_name} is "
            "missing a turns list."
        )
    return _normalize_question_id(question_id), turns


def load_mt_bench_model_answers(
    model: str,
    n_instructions: int | None = None,
    local_dir: Path | None = None,
) -> pd.DataFrame | None:
    """Load pre-generated MT-Bench answers from a local file or cached model id."""
    local_path = Path(model)
    if local_path.exists():
        answer_path = local_path
    elif "/" not in model:
        answer_path = download_mt_bench_model_answer(
            model_id=model, local_dir=local_dir
        )
    else:
        return None

    answer_records = pd.read_json(answer_path, lines=True).to_dict(orient="records")
    rows = []
    for rec in answer_records:
        question_id, turns = _extract_answer_turns(rec, str(answer_path))
        rows.append(
            {
                "instruction_index": question_id,
                "completion_turn_1": turns[0] if len(turns) > 0 else "",
                "completion_turn_2": turns[1] if len(turns) > 1 else "",
            }
        )

    df_answers = pd.DataFrame(rows)
    if df_answers.empty:
        raise ValueError(
            f"MT-Bench answer file {answer_path} did not contain any rows."
        )
    df_answers.sort_values("instruction_index", inplace=True)
    if n_instructions is not None:
        df_answers = df_answers.head(n_instructions)
    return df_answers


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
        qid = _normalize_question_id(qid_raw)

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
