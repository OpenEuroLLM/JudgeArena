from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import judgearena.generate_and_evaluate as gae
from judgearena import cache_sync
from judgearena.cache_backfill import (
    BACKFILL_PUSHED_BY,
    BackfillReport,
    backfill_sources,
    write_report,
)
from judgearena.cache_backfill_discovery import ArtifactKind, discover_sources
from judgearena.cache_backfill_sources import (
    _infer_gae_orientation,
    extract_gae_rows,
    extract_mt_bench_rows,
)
from judgearena.config import RunConfig, dump_config, meta_eval_cache_task
from judgearena.evaluate import render_judge_inputs, resolve_run_judge_prompt
from judgearena.meta_eval.cli_args import CliMetaEvalArgs
from judgearena.meta_eval.prompts import resolve_prompt_mode
from judgearena.repro import write_run_metadata
from judgearena.store_sqlite import INFERENCE_DB_NAME, SQLiteInferenceStore


def _synthetic_instructions(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {"instruction": [f"instruction {i}" for i in range(n)]},
        index=pd.Index([f"idx-{i}" for i in range(n)], name="instruction_index"),
    )


def _cfg_with_cache(tmp_path, **overrides) -> RunConfig:
    payload = {
        "task": "alpaca-eval",
        "model": {"name": "Dummy/gen-a", "baseline": "Dummy/gen-b"},
        "judge": {"model": "Dummy/score A: 0 score B: 10", "swap_mode": "fixed"},
        "generation": {"n_instructions": 2},
        "run": {"result_folder": str(tmp_path / "results"), "no_log_file": True},
        "cache": {"store_root": str(tmp_path / "live-cache")},
    }
    payload.update(overrides)
    return RunConfig(**payload)


def _write_gae_annotations(
    run_dir: Path,
    cfg: RunConfig,
    *,
    swap_both: bool = False,
) -> None:
    instructions = ["instruction 0", "instruction 1"]
    completions_a = ["completion-a-0", "completion-a-1"]
    completions_b = ["completion-b-0", "completion-b-1"]
    resolved = resolve_run_judge_prompt(cfg.task, cfg.judge)
    rendered = render_judge_inputs(
        instructions,
        completions_a,
        completions_b,
        system_prompt=resolved.system_prompt,
        user_prompt_template=resolved.user_prompt_template,
        truncate_input_chars=cfg.generation.truncate_judge_input_chars,
        provide_explanation=cfg.judge.provide_explanation,
        prompt_preset=resolved.preset_name,
        strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
        task=cfg.task,
    )
    rows = []
    for idx, (instruction, ca, cb, judge_input) in enumerate(
        zip(instructions, completions_a, completions_b, rendered, strict=True)
    ):
        rows.append(
            {
                "instruction": instruction,
                "completion_A": ca,
                "completion_B": cb,
                "judge_completion": "Score A: 8\nScore B: 6",
                "judge_input": judge_input.to_string(),
                "instruction_index": f"idx-{idx}",
                "model_A": cfg.model.name,
                "model_B": cfg.model.baseline,
                "judge": cfg.judge.model,
            }
        )
    if swap_both:
        reversed_rendered = render_judge_inputs(
            instructions,
            completions_b,
            completions_a,
            system_prompt=resolved.system_prompt,
            user_prompt_template=resolved.user_prompt_template,
            truncate_input_chars=cfg.generation.truncate_judge_input_chars,
            provide_explanation=cfg.judge.provide_explanation,
            prompt_preset=resolved.preset_name,
            strip_thinking_before_judging=cfg.judge.strip_thinking_before_judging,
            task=cfg.task,
        )
        for idx, (instruction, ca, cb, judge_input) in enumerate(
            zip(
                instructions,
                completions_b,
                completions_a,
                reversed_rendered,
                strict=True,
            )
        ):
            rows.append(
                {
                    "instruction": instruction,
                    "completion_A": ca,
                    "completion_B": cb,
                    "judge_completion": "Score A: 6\nScore B: 8",
                    "judge_input": judge_input.to_string(),
                    "instruction_index": f"idx-{idx}",
                    "model_A": cfg.model.baseline,
                    "model_B": cfg.model.name,
                    "judge": cfg.judge.model,
                }
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")
    pd.DataFrame(rows).to_csv(run_dir / "pair-annotations.csv", index=False)
    write_run_metadata(
        output_dir=run_dir,
        entrypoint="tests",
        run=cfg.model_dump(),
        results={"n": len(rows)},
        input_payloads={
            "instruction_index": [row["instruction_index"] for row in rows]
        },
        judge_system_prompt=resolved.system_prompt,
        judge_user_prompt_template=resolved.user_prompt_template,
    )


def _write_legacy_gae_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    args = {
        "task": "alpaca-eval",
        "model_A": "Dummy/gen-a",
        "model_B": "Dummy/gen-b",
        "judge_model": "Dummy/score A: 0 score B: 10",
        "swap_mode": "fixed",
        "provide_explanation": False,
        "truncate_all_input_chars": 8192,
        "engine_kwargs": {},
    }
    (run_dir / "args-alpaca.json").write_text(json.dumps(args))
    cfg = RunConfig(
        task=args["task"],
        model={"name": args["model_A"], "baseline": args["model_B"]},
        judge={"model": args["judge_model"], "swap_mode": args["swap_mode"]},
        generation={"truncate_judge_input_chars": args["truncate_all_input_chars"]},
    )
    _write_gae_annotations(run_dir, cfg)


def _count_judge_inference_rows(store_root: Path, judge_model: str) -> int:
    total = 0
    for db_path in store_root.rglob(INFERENCE_DB_NAME):
        metadata = json.loads(
            (db_path.parent / "metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("model_spec") != judge_model:
            continue
        with SQLiteInferenceStore(db_path) as store:
            total += len(store.query())
    return total


def test_discover_skips_elo_and_legacy_artifacts(tmp_path):
    elo_dir = tmp_path / "elo-lmarena-100k-run"
    elo_dir.mkdir()
    (elo_dir / "results.json").write_text("{}")

    legacy_db = tmp_path / "cache" / "db" / "arena" / "judge.db"
    legacy_db.parent.mkdir(parents=True)
    legacy_db.write_text("sqlite")

    pass_cache = tmp_path / "tables" / "model_outputs" / "alpaca-eval.csv.zip"
    pass_cache.parent.mkdir(parents=True)
    pass_cache.write_text("zip")

    report = discover_sources([tmp_path])
    skipped_kinds = {item.kind for item in report.skipped}
    assert ArtifactKind.META_EVAL_IDENTITY_DB in skipped_kinds
    assert ArtifactKind.PASS_LEVEL_CACHE in skipped_kinds
    assert ArtifactKind.ELO_RUN in skipped_kinds


def test_discover_classifies_standalone_completions_as_generation(tmp_path):
    artifact = tmp_path / "completions.parquet"
    artifact.write_bytes(b"not imported")

    report = discover_sources([artifact])

    assert [item.kind for item in report.skipped] == [ArtifactKind.GENERATION_ARTIFACT]


def test_gae_current_config_backfill_and_idempotency(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "results" / "gae-run"
    _write_gae_annotations(run_dir, cfg)

    store_root = tmp_path / "backfill-store"
    first = backfill_sources([run_dir], store_root)
    assert first.written == 2
    assert first.existing == 0

    second = backfill_sources([run_dir], store_root)
    assert second.written == 0
    assert second.existing == 2

    db_path = next(store_root.rglob(INFERENCE_DB_NAME))
    with SQLiteInferenceStore(db_path) as store:
        rows = store.query()
    assert all(row["pushed_by"] == BACKFILL_PUSHED_BY for _, row in rows.iterrows())


def test_legacy_gae_args_backfill_parity(tmp_path):
    run_dir = tmp_path / "legacy-run"
    _write_legacy_gae_run(run_dir)
    store_root = tmp_path / "store"
    report = backfill_sources([run_dir], store_root)
    assert report.written == 2
    assert report.sources["gae"]["runs"] == 1


def test_gae_swap_rows_backfill(tmp_path):
    cfg = _cfg_with_cache(
        tmp_path,
        judge={"model": "Dummy/score A: 0 score B: 10", "swap_mode": "both"},
    )
    run_dir = tmp_path / "swap-run"
    _write_gae_annotations(run_dir, cfg, swap_both=True)

    report = backfill_sources([run_dir], tmp_path / "store")
    assert report.written == 4


def test_gae_judge_input_mismatch_skipped(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "mismatch-run"
    _write_gae_annotations(run_dir, cfg)
    csv_path = run_dir / "pair-annotations.csv"
    df = pd.read_csv(csv_path)
    df.loc[0, "judge_input"] = "tampered"
    df.to_csv(csv_path, index=False)

    report = backfill_sources([run_dir], tmp_path / "store")
    assert report.written == 1
    assert report.skipped["judge_input_mismatch"] == 1


def test_gae_missing_judge_output_is_not_backfilled(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "missing-output"
    _write_gae_annotations(run_dir, cfg)
    csv_path = run_dir / "pair-annotations.csv"
    df = pd.read_csv(csv_path, keep_default_na=False)
    df.loc[0, "judge_completion"] = ""
    df.to_csv(csv_path, index=False)

    report = backfill_sources([run_dir], tmp_path / "store")

    assert report.written == 1
    assert report.skipped["judge_output_missing"] == 1


def test_multiple_annotations_files_fail_closed(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "multiple-annotations"
    _write_gae_annotations(run_dir, cfg)
    (run_dir / "other-annotations.csv").write_bytes(
        (run_dir / "pair-annotations.csv").read_bytes()
    )

    report = backfill_sources([run_dir], tmp_path / "store")

    assert report.written == 0
    assert report.skipped["source_extraction_failed"] == 1


def test_conflicting_modern_configs_fail_closed(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "conflicting-configs"
    _write_gae_annotations(run_dir, cfg)
    conflicting = cfg.model_copy(deep=True)
    conflicting.judge.model = "Dummy/different-judge"
    dump_config(conflicting, run_dir / "config.yaml")

    report = backfill_sources([run_dir], tmp_path / "store")

    assert report.written == 0
    assert report.skipped["source_extraction_failed"] == 1


def test_gae_ambiguous_orientation_is_skipped(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "ambiguous-orientation"
    _write_gae_annotations(run_dir, cfg)
    csv_path = run_dir / "pair-annotations.csv"
    df = pd.read_csv(csv_path)
    df.loc[0, ["model_A", "model_B"]] = ["other-a", "other-b"]
    df.to_csv(csv_path, index=False)

    report = backfill_sources([run_dir], tmp_path / "store")

    assert report.written == 1
    assert report.skipped["battle_orientation_unverifiable"] == 1


def test_gae_missing_judge_input_fail_closed(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "missing-input-run"
    _write_gae_annotations(run_dir, cfg)
    csv_path = run_dir / "pair-annotations.csv"
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["judge_input"])
    df.to_csv(csv_path, index=False)

    report = backfill_sources([run_dir], tmp_path / "store")
    assert report.written == 0
    assert report.skipped.get("unknown_judge_run", 0) == 1


def test_local_engine_skipped_without_init(tmp_path, monkeypatch):
    cfg = _cfg_with_cache(
        tmp_path,
        judge={"model": "VLLM/Qwen/Qwen2.5-0.5B-Instruct", "swap_mode": "fixed"},
    )
    run_dir = tmp_path / "vllm-run"
    _write_gae_annotations(run_dir, cfg)

    def fail_init(*args, **kwargs):
        raise AssertionError("VLLM should not initialize during backfill")

    monkeypatch.setattr(
        "judgearena.models.ChatVLLM.__init__",
        fail_init,
    )

    report = backfill_sources([run_dir], tmp_path / "store")
    assert report.written == 0
    assert report.skipped["local_engine_unsupported"] == 2


def test_mt_preset_backfill(tmp_path):
    cfg = _cfg_with_cache(
        tmp_path,
        task="mt-bench",
        model={"name": "Dummy/a", "baseline": "Dummy/b"},
        judge={"model": "Dummy/judge-output", "swap_mode": "fixed"},
    )
    run_dir = tmp_path / "mt-preset"
    run_dir.mkdir(parents=True)
    dump_config(cfg, run_dir / "config.yaml")
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "category": "writing",
                "turn": 1,
                "model_A": "Dummy/a",
                "model_B": "Dummy/b",
                "judge": "Dummy/judge-output",
                "prompt_name": "default-single",
                "system_prompt": "system",
                "user_prompt": "user body",
                "judge_completion": "Score A: 8\nScore B: 6",
                "swapped": False,
            }
        ]
    ).to_csv(run_dir / "mt-annotations.csv", index=False)

    report = backfill_sources([run_dir], tmp_path / "store")
    assert report.written == 1
    assert report.sources["mt_bench"]["runs"] == 1


def test_mt_fastchat_g1_g2_backfill(tmp_path):
    cfg = _cfg_with_cache(
        tmp_path,
        task="mt-bench",
        model={"name": "Dummy/a", "baseline": "Dummy/b"},
        judge={"model": "Dummy/judge-output", "swap_mode": "both"},
    )
    run_dir = tmp_path / "mt-fastchat"
    run_dir.mkdir(parents=True)
    dump_config(cfg, run_dir / "config.yaml")
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "category": "writing",
                "turn": 1,
                "model_A": "Dummy/a",
                "model_B": "Dummy/b",
                "judge": "Dummy/judge-output",
                "prompt_name": "pair-v2",
                "system_prompt": "system",
                "g1_user_prompt": "  g1 user  ",
                "g1_judgment": " [[A]] ",
                "g2_user_prompt": "g2 user",
                "g2_judgment": "[[B]]",
            }
        ]
    ).to_csv(run_dir / "mt-annotations.csv", index=False)

    report = backfill_sources([run_dir], tmp_path / "store")
    assert report.written == 2
    extraction = extract_mt_bench_rows(run_dir)
    assert "  g1 user  " in extraction.rows[0].canonical_input
    assert extraction.rows[0].output_text == " [[A]] "


def test_meta_eval_forward_and_swapped(tmp_path):
    args = CliMetaEvalArgs(
        judge_model="Dummy/meta-judge",
        reference_arena="LMArena-140k",
        prompt_mode="standard",
        swap_mode="both",
    )
    prompt_spec = resolve_prompt_mode(args.prompt_mode, provide_explanation=False)
    forward_rendered = render_judge_inputs(
        ["instruction"],
        ["completion a"],
        ["completion b"],
        system_prompt=prompt_spec.system_prompt,
        user_prompt_template=prompt_spec.user_prompt_template,
        truncate_input_chars=args.truncate_judge_input_chars,
        provide_explanation=False,
    )[0].to_string()
    swapped_rendered = render_judge_inputs(
        ["instruction"],
        ["completion b"],
        ["completion a"],
        system_prompt=prompt_spec.system_prompt,
        user_prompt_template=prompt_spec.user_prompt_template,
        truncate_input_chars=args.truncate_judge_input_chars,
        provide_explanation=False,
    )[0].to_string()

    run_dir = tmp_path / "meta-run"
    run_dir.mkdir(parents=True)
    (run_dir / "args.json").write_text(json.dumps(args.to_jsonable()))
    pd.DataFrame(
        [
            {
                "question_id": "q1",
                "benchmark": "arena",
                "model_a": "m-a",
                "model_b": "m-b",
                "instruction": "instruction",
                "completion_a": "completion a",
                "completion_b": "completion b",
                "presented_completion_a": "completion a",
                "presented_completion_b": "completion b",
                "judge_input": forward_rendered,
                "judge_completion": "Score A: 8\nScore B: 6",
                "orientation": "forward",
                "presented_model_a": "m-a",
                "presented_model_b": "m-b",
            },
            {
                "question_id": "q1",
                "benchmark": "arena",
                "model_a": "m-a",
                "model_b": "m-b",
                "instruction": "instruction",
                "completion_a": "completion a",
                "completion_b": "completion b",
                "presented_completion_a": "completion b",
                "presented_completion_b": "completion a",
                "judge_input": swapped_rendered,
                "judge_completion": "Score A: 6\nScore B: 8",
                "orientation": "swapped",
                "presented_model_a": "m-b",
                "presented_model_b": "m-a",
            },
        ]
    ).to_parquet(run_dir / "annotations.parquet")

    report = backfill_sources([run_dir], tmp_path / "store")
    assert report.written == 2
    assert report.rows_planned == 2
    task = meta_eval_cache_task(args.reference_arena)
    assert any(task in str(path) for path in (tmp_path / "store").rglob("*"))
    db_path = next((tmp_path / "store").rglob(INFERENCE_DB_NAME))
    with SQLiteInferenceStore(db_path) as store:
        meta_rows = store.query_metadata()
    assert all(
        "source_run_id" in json.loads(row["metadata_json"])
        for _, row in meta_rows.iterrows()
    )
    assert all(
        "source_run_folder" not in json.loads(row["metadata_json"])
        for _, row in meta_rows.iterrows()
    )


def test_conflicting_outputs_skipped(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _write_gae_annotations(run_a, cfg)
    _write_gae_annotations(run_b, cfg)
    df = pd.read_csv(run_b / "pair-annotations.csv")
    df.loc[0, "judge_completion"] = "Score A: 1\nScore B: 9"
    df.to_csv(run_b / "pair-annotations.csv", index=False)

    report = backfill_sources([run_a, run_b], tmp_path / "store")
    assert report.skipped.get("conflicting_outputs", 0) == 2
    assert _count_judge_inference_rows(tmp_path / "store", cfg.judge.model) <= 2


def test_identical_outputs_preserve_all_run_metadata(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    _write_gae_annotations(run_a, cfg)
    _write_gae_annotations(run_b, cfg)

    store_root = tmp_path / "store"
    report = backfill_sources([run_a, run_b], store_root)

    assert report.written == 2
    db_path = next(store_root.rglob(INFERENCE_DB_NAME))
    with SQLiteInferenceStore(db_path) as store:
        metadata = store.query_metadata()
    source_ids = {
        json.loads(value)["source_run_id"] for value in metadata["metadata_json"]
    }
    assert source_ids == {"run-a", "run-b"}
    assert len(metadata) == 4


def test_dry_run_writes_nothing(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "dry-run"
    _write_gae_annotations(run_dir, cfg)
    report = backfill_sources([run_dir], tmp_path / "store", dry_run=True)
    assert report.written == 2
    assert _count_judge_inference_rows(tmp_path / "store", cfg.judge.model) == 0


def test_backfill_cli(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "cli-run"
    _write_gae_annotations(run_dir, cfg)
    report_path = tmp_path / "report.json"

    cache_sync.main(
        [
            "backfill",
            str(run_dir),
            "--store_root",
            str(tmp_path / "store"),
            "--report",
            str(report_path),
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["written"] == 2
    assert _count_judge_inference_rows(tmp_path / "store", cfg.judge.model) == 2


@pytest.fixture
def mock_gae_inputs(monkeypatch):
    monkeypatch.setattr(
        gae,
        "load_instructions",
        lambda dataset, n_instructions=None: _synthetic_instructions(
            n_instructions or 2
        ),
    )
    monkeypatch.setattr(gae, "try_load_dataset_completions", lambda *args: None)


def test_gae_live_run_backfill_reuses_cells(mock_gae_inputs, tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    gae.main(cfg)
    run_dir = next((tmp_path / "results").iterdir())

    live_count = _count_judge_inference_rows(tmp_path / "live-cache", cfg.judge.model)
    backfill_store = tmp_path / "backfill-store"
    report = backfill_sources([run_dir], backfill_store)
    assert report.written == live_count
    assert _count_judge_inference_rows(backfill_store, cfg.judge.model) == live_count


def test_write_report_roundtrip(tmp_path):
    report = BackfillReport(
        written=3,
        existing=1,
        rows_planned=4,
        skipped={"judge_input_mismatch": 2},
    )
    path = tmp_path / "nested" / "report.json"
    write_report(report, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["written"] == 3
    assert payload["rows_planned"] == 4
    assert payload["skipped"]["judge_input_mismatch"] == 2


def test_gae_orientation_inference_reversed_rows():
    cfg = RunConfig(
        task="alpaca-eval",
        model={"name": "Dummy/gen-a", "baseline": "Dummy/gen-b"},
        judge={"model": "Dummy/judge"},
    )
    direct = pd.Series({"model_A": "Dummy/gen-a", "model_B": "Dummy/gen-b"})
    reversed_row = pd.Series({"model_A": "Dummy/gen-b", "model_B": "Dummy/gen-a"})
    assert _infer_gae_orientation(direct, cfg=cfg) == "direct"
    assert _infer_gae_orientation(reversed_row, cfg=cfg) == "reversed"


def test_source_extraction_failure_continues_other_runs(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    good_run = tmp_path / "good-run"
    bad_run = tmp_path / "bad-run"
    _write_gae_annotations(good_run, cfg)
    bad_run.mkdir()
    pd.DataFrame(
        {
            "instruction": ["instruction 0"],
            "completion_A": ["a"],
            "completion_B": ["b"],
            "judge_input": ["prompt"],
            "judge_completion": ["Score A: 1\nScore B: 0"],
        }
    ).to_csv(bad_run / "pair-annotations.csv", index=False)

    report = backfill_sources([bad_run, good_run], tmp_path / "store")
    assert report.skipped["source_extraction_failed"] == 1
    assert report.written == 2


def test_unknown_annotation_csv_skipped_not_migrated(tmp_path):
    run_dir = tmp_path / "custom-run"
    run_dir.mkdir()
    pd.DataFrame([{"foo": 1, "bar": 2}]).to_csv(
        run_dir / "custom-annotations.csv", index=False
    )
    report = discover_sources([run_dir])
    assert report.migratable_runs == []
    assert any(item.kind == ArtifactKind.UNKNOWN for item in report.skipped)


def test_conflicting_existing_output_skips_metadata(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "gae-run"
    _write_gae_annotations(run_dir, cfg)

    store_root = tmp_path / "store"
    first = backfill_sources([run_dir], store_root)
    assert first.written == 2

    df = pd.read_csv(run_dir / "pair-annotations.csv")
    df.loc[0, "judge_completion"] = "Score A: 1\nScore B: 9"
    df.to_csv(run_dir / "pair-annotations.csv", index=False)

    second = backfill_sources([run_dir], store_root)
    assert second.skipped.get("conflicting_existing_output", 0) == 1
    db_path = next(store_root.rglob(INFERENCE_DB_NAME))
    with SQLiteInferenceStore(db_path) as store:
        rows = store.query()
    assert rows.iloc[0]["output_text"] == "Score A: 8\nScore B: 6"


def test_legacy_multiple_args_files_fail_closed(tmp_path):
    run_dir = tmp_path / "ambiguous-legacy"
    run_dir.mkdir()
    (run_dir / "args-a.json").write_text(json.dumps({"task": "alpaca-eval"}))
    (run_dir / "args-b.json").write_text(json.dumps({"task": "alpaca-eval"}))
    pd.DataFrame(
        {
            "instruction": ["x"],
            "completion_A": ["a"],
            "completion_B": ["b"],
            "judge_input": ["prompt"],
            "judge_completion": ["Score A: 1\nScore B: 0"],
        }
    ).to_csv(run_dir / "pair-annotations.csv", index=False)

    report = backfill_sources([run_dir], tmp_path / "store")
    assert report.skipped["source_extraction_failed"] == 1
    assert report.written == 0


def test_legacy_truncate_all_input_chars_used_for_judge(tmp_path):
    run_dir = tmp_path / "legacy-truncate"
    _write_legacy_gae_run(run_dir)
    extraction = extract_gae_rows(run_dir)
    assert extraction.rows
    assert extraction.skipped.get("judge_input_mismatch", 0) == 0


def test_nested_legacy_db_does_not_block_parent_run_discovery(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "results" / "nested-run"
    _write_gae_annotations(run_dir, cfg)
    legacy_dir = tmp_path / "results" / "cache" / "nested"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "judgements.db").write_text("sqlite")

    report = discover_sources([tmp_path / "results"])
    migratable = {item.run_dir.name for item in report.migratable_runs}
    assert "nested-run" in migratable
    skipped_kinds = {item.kind for item in report.skipped}
    assert ArtifactKind.LEGACY_CACHE_CELL in skipped_kinds

    backfill_report = backfill_sources([tmp_path / "results"], tmp_path / "store")
    assert backfill_report.written == 2


def test_mt_swapped_string_and_nan_prompts(tmp_path):
    cfg = _cfg_with_cache(
        tmp_path,
        task="mt-bench",
        model={"name": "Dummy/a", "baseline": "Dummy/b"},
        judge={"model": "Dummy/judge-output", "swap_mode": "fixed"},
    )
    run_dir = tmp_path / "mt-swapped-string"
    run_dir.mkdir(parents=True)
    dump_config(cfg, run_dir / "config.yaml")
    pd.DataFrame(
        [
            {
                "question_id": 1,
                "category": "writing",
                "turn": 1,
                "model_A": "Dummy/a",
                "model_B": "Dummy/b",
                "judge": "Dummy/judge-output",
                "prompt_name": "default-single",
                "system_prompt": float("nan"),
                "user_prompt": "user body",
                "judge_completion": "Score A: 8\nScore B: 6",
                "swapped": "true",
            }
        ]
    ).to_csv(run_dir / "mt-annotations.csv", index=False)

    report = backfill_sources([run_dir], tmp_path / "store")
    assert report.written == 1
    db_path = next((tmp_path / "store").rglob(INFERENCE_DB_NAME))
    with SQLiteInferenceStore(db_path) as store:
        row = store.query().iloc[0]
    assert "nan" not in row["input_text"].lower()


def test_dry_run_does_not_touch_existing_db(tmp_path):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "dry-run-existing"
    _write_gae_annotations(run_dir, cfg)
    store_root = tmp_path / "store"
    backfill_sources([run_dir], store_root, dry_run=False)
    db_path = next(store_root.rglob(INFERENCE_DB_NAME))
    before = db_path.stat().st_mtime_ns
    wal_path = Path(f"{db_path}-wal")
    wal_before_exists = wal_path.exists()

    report = backfill_sources([run_dir], store_root, dry_run=True)
    assert report.written == 0
    assert report.existing == 2
    assert db_path.stat().st_mtime_ns == before
    assert wal_path.exists() == wal_before_exists


def test_cell_integrity_error_is_reported(tmp_path, monkeypatch):
    cfg = _cfg_with_cache(tmp_path)
    run_dir = tmp_path / "integrity-run"
    _write_gae_annotations(run_dir, cfg)

    def fail_metadata(*args, **kwargs):
        raise ValueError("metadata mismatch")

    monkeypatch.setattr(
        "judgearena.cache_backfill.write_store_metadata",
        fail_metadata,
    )
    report = backfill_sources([run_dir], tmp_path / "store")
    assert report.skipped.get("cell_integrity_error", 0) == 2
    assert report.written == 0
