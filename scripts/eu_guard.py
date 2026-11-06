"""Classify EU-Guard prompts with an external Llama Guard model.

python scripts/eu_guard.py --model meta-llama/Llama-Guard-3-8B
Llama Guard uses its own policy, which differs from EU-Guard's.
For Jev and chat judges, use the eu-guard-guards task through judgearena.
"""

import argparse
import json
from pathlib import Path

from judgearena.artifacts import write_annotations
from judgearena.benchmarks.scoring import build_metrics, calculate_metrics
from judgearena.datasets import load_task_data, resolve_dataset_filters
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Llama Guard model")
    parser.add_argument("--task", default="eu-guard")
    parser.add_argument(
        "--language", help="Optional language filter; task variants also apply"
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/eu-guard-prompts"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
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
    annotations = judge_llama_guard(rows, args.model)
    annotations["judge"] = args.model
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
            },
            default=str,
            indent=2,
        )
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
