"""Dataset adapter for the YAML-defined MT-Bench task."""

from __future__ import annotations

import warnings
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from huggingface_hub import snapshot_download

from judgearena.paths import data_root
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import (
    GitRawSource,
    HuggingFaceSpaceSource,
    ResolvedTaskSpec,
    TaskDefaultBaseline,
)


def _task(task_id: str = "mt-bench") -> ResolvedTaskSpec:
    task = get_packaged_task(task_id)
    if task is None or task.spec.dataset.adapter != "mt_bench":
        raise ValueError(f"Unsupported MT-Bench task: {task_id!r}.")
    return task


def _space_source(task: ResolvedTaskSpec) -> HuggingFaceSpaceSource:
    source = task.spec.dataset.sources.get("benchmark")
    if not isinstance(source, HuggingFaceSpaceSource):
        raise ValueError(
            f"Task {task.task!r} must define a Hugging Face Space source "
            "named 'benchmark'."
        )
    return source


def _reference_source(task: ResolvedTaskSpec) -> GitRawSource:
    source = task.spec.dataset.sources.get("references")
    if not isinstance(source, GitRawSource):
        raise ValueError(
            f"Task {task.task!r} must define a Git raw source named 'references'."
        )
    return source


def _task_cache_dir(task: ResolvedTaskSpec, local_tables_path: Path) -> Path:
    return local_tables_path / "_sources" / task.definition_task


def _normalize_question_id(question_id: object) -> object:
    try:
        return int(question_id)
    except Exception:
        return question_id


def _snapshot_mt_bench_files(
    *,
    task: ResolvedTaskSpec,
    local_dir: Path,
    allow_patterns: list[str],
    expected_path: Path,
    description: str,
) -> None:
    source = _space_source(task)
    try:
        snapshot_download(
            repo_id=source.repo_id,
            repo_type="space",
            allow_patterns=allow_patterns,
            local_dir=local_dir,
            force_download=False,
            revision=source.revision,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {description} from Hugging Face Space "
            f"{source.repo_id!r}. If you are offline, place the file at "
            f"{expected_path}."
        ) from exc
    if not expected_path.exists():
        raise FileNotFoundError(
            f"Could not locate {description} after download. Expected {expected_path}."
        )


def _git_raw_url(source: GitRawSource) -> str:
    repository = source.repository.rstrip("/")
    github_prefix = "https://github.com/"
    if repository.startswith(github_prefix):
        project = repository.removeprefix(github_prefix)
        return (
            f"https://raw.githubusercontent.com/{project}/{source.revision}/"
            f"{source.path}"
        )
    return f"{repository}/raw/{source.revision}/{source.path}"


def _download_references(task: ResolvedTaskSpec, local_dir: Path) -> Path | None:
    reference_dir = local_dir / "reference_answer"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_path = reference_dir / "gpt-4.jsonl"
    if reference_path.exists():
        return reference_path
    source = _reference_source(task)
    try:
        urlretrieve(_git_raw_url(source), reference_path)
    except Exception as exc:
        warnings.warn(
            "Could not download MT-Bench GPT-4 reference answers. Falling back "
            f"to inline references from question.jsonl: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return reference_path


def _download_mt_bench(
    task: ResolvedTaskSpec, local_dir: Path
) -> tuple[Path, Path | None]:
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create MT-Bench cache directory at {local_dir}. Set "
            "JUDGEARENA_DATA to a writable location."
        ) from exc

    question_path = local_dir / "data" / "mt_bench" / "question.jsonl"
    if not question_path.exists():
        _snapshot_mt_bench_files(
            task=task,
            local_dir=local_dir,
            allow_patterns=[question_path.relative_to(local_dir).as_posix()],
            expected_path=question_path,
            description="MT-Bench questions",
        )
    return question_path, _download_references(task, local_dir)


def download_mt_bench(local_dir: Path | None = None) -> tuple[Path, Path | None]:
    """Compatibility wrapper downloading the registered MT-Bench sources."""
    return _download_mt_bench(_task(), local_dir or data_root / "mt-bench")


def download_mt_bench_model_answer(
    model_id: str,
    local_dir: Path | None = None,
    *,
    task: ResolvedTaskSpec | None = None,
) -> Path:
    """Download a cached MT-Bench model-answer file if missing."""
    resolved = task or _task()
    root = local_dir or data_root / "mt-bench"
    answer_path = root / "data" / "mt_bench" / "model_answer" / f"{model_id}.jsonl"
    if answer_path.exists():
        return answer_path
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    _snapshot_mt_bench_files(
        task=resolved,
        local_dir=root,
        allow_patterns=[answer_path.relative_to(root).as_posix()],
        expected_path=answer_path,
        description=f"MT-Bench model answers for {model_id!r}",
    )
    return answer_path


def download_task_sources(task: ResolvedTaskSpec, local_tables_path: Path) -> None:
    """Download every source required by the registered MT-Bench task."""
    if task.spec.dataset.adapter != "mt_bench":
        raise ValueError(f"Task {task.task!r} does not use the MT-Bench adapter.")
    local_dir = _task_cache_dir(task, local_tables_path)
    _download_mt_bench(task, local_dir)
    baseline = task.spec.protocol.baseline
    if isinstance(baseline, TaskDefaultBaseline):
        download_mt_bench_model_answer(
            baseline.reference_id,
            local_dir=local_dir,
            task=task,
        )


def _extract_answer_turns(record: dict, source_name: str) -> tuple[object, list[str]]:
    question_id = record.get("question_id", record.get("id"))
    if question_id is None:
        raise ValueError(
            f"MT-Bench answer record from {source_name} is missing question_id/id."
        )
    choices = record.get("choices")
    if not (isinstance(choices, list) and choices):
        raise ValueError(
            f"MT-Bench answer record for question {question_id} in {source_name} "
            "is missing a non-empty choices list."
        )
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError(
            f"MT-Bench answer record for question {question_id} in {source_name} "
            "has a malformed first choice entry."
        )
    turns = first_choice.get("turns")
    if not isinstance(turns, list):
        raise ValueError(
            f"MT-Bench answer record for question {question_id} in {source_name} "
            "is missing a turns list."
        )
    return _normalize_question_id(question_id), turns


def load_mt_bench_model_answers(
    model: str,
    n_instructions: int | None = None,
    local_dir: Path | None = None,
    *,
    task: ResolvedTaskSpec | None = None,
) -> pd.DataFrame | None:
    """Load pre-generated MT-Bench answers from a path or cached model ID."""
    local_path = Path(model)
    if local_path.exists():
        answer_path = local_path
    elif "/" not in model:
        answer_path = download_mt_bench_model_answer(
            model_id=model,
            local_dir=local_dir,
            task=task,
        )
    else:
        return None

    answer_records = pd.read_json(answer_path, lines=True).to_dict(orient="records")
    rows = []
    for record in answer_records:
        question_id, turns = _extract_answer_turns(record, str(answer_path))
        rows.append(
            {
                "instruction_index": question_id,
                "completion_turn_1": turns[0] if turns else "",
                "completion_turn_2": turns[1] if len(turns) > 1 else "",
            }
        )

    df_answers = pd.DataFrame(rows)
    if df_answers.empty:
        raise ValueError(f"MT-Bench answer file {answer_path} contained no rows.")
    df_answers.sort_values("instruction_index", inplace=True)
    return df_answers.head(n_instructions) if n_instructions is not None else df_answers


def _load_mt_bench(task: ResolvedTaskSpec, local_dir: Path) -> pd.DataFrame:
    question_path, reference_path = _download_mt_bench(task, local_dir)
    questions = pd.read_json(question_path, lines=True).to_dict(orient="records")

    references_by_id: dict[int | str, list[str]] = {}
    use_inline_references = reference_path is None
    if reference_path is not None:
        try:
            records = pd.read_json(reference_path, lines=True).to_dict(orient="records")
            for record in records:
                question_id = record.get("question_id", record.get("id"))
                choices = record.get("choices")
                if question_id is None or not (isinstance(choices, list) and choices):
                    continue
                first_choice = choices[0]
                turns = (
                    first_choice.get("turns")
                    if isinstance(first_choice, dict)
                    else None
                )
                if not isinstance(turns, list):
                    continue
                references_by_id[question_id] = turns
                references_by_id[_normalize_question_id(question_id)] = turns
        except Exception as exc:
            warnings.warn(
                "Failed to parse MT-Bench GPT-4 references. Falling back to "
                f"inline references from question.jsonl: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            use_inline_references = True

    fields = task.spec.dataset.fields
    rows = []
    for record in questions:
        question_id_raw = record.get(fields.id, record.get("id"))
        if question_id_raw is None:
            raise ValueError(
                f"MT-Bench question is missing field {fields.id!r}: keys={list(record)}"
            )
        question_id = _normalize_question_id(question_id_raw)
        turns = record.get(fields.instruction)
        if isinstance(turns, list):
            turn_1 = turns[0] if turns else None
            turn_2 = turns[1] if len(turns) > 1 else None
        else:
            turn_1 = turns
            turn_2 = record.get("turn_2")

        reference_turns = references_by_id.get(question_id_raw) or references_by_id.get(
            question_id
        )
        if reference_turns is None and use_inline_references:
            inline_reference = record.get("reference")
            if isinstance(inline_reference, list):
                reference_turns = inline_reference

        rows.append(
            {
                "instruction_index": question_id,
                "category": (
                    record.get(fields.category) if fields.category is not None else None
                ),
                "turn_1": turn_1,
                "turn_2": turn_2,
                "reference_turn_1": (reference_turns[0] if reference_turns else None),
                "reference_turn_2": (
                    reference_turns[1]
                    if reference_turns is not None and len(reference_turns) > 1
                    else None
                ),
                "instruction": turn_1,
            }
        )
    return pd.DataFrame(rows)


def load_task_instructions(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame:
    """Load normalized MT-Bench questions through the dataset registry."""
    return _load_mt_bench(task, _task_cache_dir(task, local_tables_path))


def load_task_model_outputs(
    task: ResolvedTaskSpec, local_tables_path: Path
) -> pd.DataFrame | None:
    """MT-Bench loads two-turn model answers through its specialized runner."""
    return None


def load_mt_bench() -> pd.DataFrame:
    """Compatibility wrapper loading the registered MT-Bench task."""
    return _load_mt_bench(_task(), data_root / "mt-bench")
