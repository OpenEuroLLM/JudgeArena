"""Classify EU-Guard prompts directly; no candidate answers are generated.

Jev: python scripts/eu_guard.py --config configs/eu_guard_prompt_jev.yaml
Local guard: python scripts/eu_guard.py --llama-guard --model meta-llama/Llama-Guard-3-8B
Jev uses an example rubric, not the official EU-Guard evaluator.
The local example uses Llama Guard's own policy, which differs from EU-Guard's.
"""

import argparse
import json
from pathlib import Path

import yaml

from judgearena.artifacts import write_annotations
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.cache.inference import JudgementInferenceCache
from judgearena.config import JudgeArgs
from judgearena.datasets import load_task_data, resolve_dataset_filters
from judgearena.judging import JudgeRequest, annotation_frame, execute_judging
from judgearena.models import prepare_model
from judgearena.prompts.registry import render_inputs, resolve_run_judge_prompt
from judgearena.tasks.registry import get_packaged_task


def parse_llama_guard(text):
    """Keep the guard's categories; unfamiliar outputs remain missing."""
    lines = text.strip().splitlines()
    label = {"safe": "answer", "unsafe": "refuse"}.get(lines[0] if lines else "")
    return {"label": label, "details": {"categories": lines[1:]}} if label else None


def judge_llama_guard(rows, model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", dtype="auto"
    )
    model.eval()
    raw, inputs = [], []
    for instruction in rows["instruction"]:
        messages = [{"role": "user", "content": instruction}]
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=64, do_sample=False)
        raw.append(
            tokenizer.decode(
                output[0, encoded["input_ids"].shape[-1] :], skip_special_tokens=True
            )
        )
        inputs.append(
            tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
        )
    return rows.assign(
        instruction_index=rows.index,
        judge_input=inputs,
        judge_completion=raw,
        parsed=[parse_llama_guard(text) for text in raw],
    )


def judge_jev(rows, judge_cfg, cache):
    prompt = resolve_run_judge_prompt(None, judge_cfg)
    requests = [
        JudgeRequest(
            str(i),
            value,
            prompt.parser,
            {"instruction_id": str(i), "model_a": judge_cfg.model, "model_b": None},
        )
        for i, value in zip(
            rows.index, render_inputs(prompt, rows.to_dict("records")), strict=True
        )
    ]
    model = prepare_model(judge_cfg.model, cache=cache, **judge_cfg.model_kwargs())
    results = execute_judging(model, requests, use_tqdm=True)
    return annotation_frame(rows, results, [prompt] * len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eu_guard_prompt_jev.yaml")
    parser.add_argument(
        "--model", help="Override the configured judge or select a local guard"
    )
    parser.add_argument("--llama-guard", action="store_true")
    parser.add_argument("--task", default="eu-guard")
    parser.add_argument(
        "--language", help="Optional language filter; task variants also apply"
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/eu-guard-prompts"))
    parser.add_argument("--cache", type=Path, default=Path("cache/eu-guard-prompts"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.llama_guard and not args.model:
        parser.error("--llama-guard requires --model")
    judge_cfg = None
    if not args.llama_guard:
        with open(args.config) as f:
            judge_cfg = JudgeArgs.model_validate(yaml.safe_load(f)["judge"])
        if args.model:
            judge_cfg.model = args.model
    model_name = args.model if args.llama_guard else judge_cfg.model
    task = get_packaged_task(args.task)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = (
        load_task_data(task)
        .select(
            resolve_dataset_filters(
                task, {"lang": [args.language]} if args.language else {}
            )
        )
        .instructions.head(args.limit)
    )
    annotations = (
        judge_llama_guard(rows, args.model)
        if args.llama_guard
        else judge_jev(rows, judge_cfg, JudgementInferenceCache(args.cache, task.task))
    )
    annotations["judge"] = model_name
    metrics = calculate_metrics(
        annotations, build_metrics(task.spec.protocol.scoring.metrics)
    )
    write_annotations(annotations, args.output / "annotations.parquet")
    (args.output / "results.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False)
    )
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "task": task.spec.model_dump(mode="json"),
                "judge": judge_cfg.model_dump(mode="json")
                if judge_cfg is not None
                else None,
            },
            default=str,
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
