"""Freeze the approved intermediate experiment; no inference."""

import hashlib
import json
from pathlib import Path
import yaml

from judgearena.config import RunConfig
from judgearena.arenas_utils import extract_turn_text
from judgearena.datasets import load_battles
from judgearena.tasks.registry import get_packaged_task
from judgearena.benchmarks.meta_eval.runner import _prepare_arena_battles
from judgearena.benchmarks.meta_eval.sampling import (
    filter_split,
    select_top_models,
    comparison_components,
)

LOCAL = Path("/private/tmp/judgearena-tuning-intermediate-20261014")
REMOTE = Path("/work/dlclarge1/lushtake-hiwi/judge-tuning-intermediate-20261014")
OLD = Path("/private/tmp/judgearena-tuning-v2")
source = LOCAL / "source"
templates = source / "judgearena/prompts/templates"
original = (templates / "prompt.txt").read_text()
start = original.split("# Your output")[0]
guides = {
    "neutral": "",
    "criteria": "Apply the same instruction-relevant criteria to both assistants: correctness, helpfulness, completeness and clarity where applicable. Equal quality on a criterion must receive equal treatment. Do not invent unrelated criteria.",
    "evidence": "Evaluate concrete supporting claims, errors and omissions in each answer. Base the judgment on observable evidence rather than vague preference or style.",
    "verification": "Check whether each answer satisfies the instruction, then check factual errors, contradictions and logical flaws. Penalize failed requirements and clear errors more heavily than style or length.",
    "verdict": "Evaluate correctness, instruction satisfaction, helpfulness and completeness consistently for both answers.",
}
entries = []
meta = []
for guide, wording in guides.items():
    for explanation in [False, True]:
        if guide == "neutral":
            name = "default_with_explanation" if explanation else "default"
            filename = "prompt-with-explanation.txt" if explanation else "prompt.txt"
        else:
            name = f"mid-{guide}-" + ("on" if explanation else "off")
            filename = name + ".txt"
            instruction = (
                "First give a brief justification, then return the final scores."
                if explanation
                else "Return only the final scores, without a written explanation."
            )
            if guide == "verdict":
                formatting = "Return exactly one final verdict: [[A>>B]], [[A>B]], [[A=B]], [[B>A]], or [[B>>A]]. Use >> for a substantial quality difference, > for a smaller difference, and = for a tie."
                instruction = instruction.replace("scores", "verdict")
            else:
                formatting = "Return the final scores as two lines:\nscore_A: <an integer from 0 to 10>\nscore_B: <an integer from 0 to 10>"
            (templates / filename).write_text(
                start
                + "# Evaluation guidance\n"
                + wording
                + "\n\n# Your output\n"
                + formatting
                + "\n"
                + instruction
                + "\n"
            )
            parser = (
                'JUDGE_PARSERS["arena-hard-verdict"]'
                if guide == "verdict"
                else "SCORE_PARSER"
            )
            entries.append(
                f'    "{name}": JudgePromptPreset(name="{name}", parser={parser}, system_file="system-prompt.txt", user_file="{filename}"),'
            )
        meta.append(
            {
                "preset": name,
                "base_prompt": guide,
                "explanation": explanation,
                "template_sha256": hashlib.sha256(
                    (templates / filename).read_bytes()
                ).hexdigest(),
            }
        )
p = source / "judgearena/prompts/registry.py"
s = p.read_text()
s = s.replace(
    "JUDGE_PROMPT_PRESETS = tuple(PRESETS)",
    "PRESETS.update({\n"
    + "\n".join(entries)
    + "\n})\n\nJUDGE_PROMPT_PRESETS = tuple(PRESETS)",
)
p.write_text(s)
manifest = json.loads((OLD / "configs/stage0-manifest.json").read_text())
manifest.update(
    stage="Intermediate approved;120allocatedL40S-hours;8GPUpeak;noAPI/heldout",
    presets=[m["preset"] for m in meta],
    prompt_factors=meta,
    cohort=[],
    optimizer_encoding="10 categorical presets enumerate the 5x2 Cartesian product",
    total_l40s_hour_cap=120,
    source_base_commit="ab7cc3f5c22527ba29ed2d337c92e2f689bc25b4",
)
task = get_packaged_task("meta-eval-lmarena-140k-en")
arena = _prepare_arena_battles(
    load_battles(task), task=task.task, arena="LMArena-140k", languages=["en"]
)
models, pool = select_top_models(arena, top_models=20)
validation = filter_split(
    pool,
    pool.conversation_a.map(lambda x: extract_turn_text(x[0])),
    split="validation",
    validation_fraction=0.5,
).set_index("battle_id")
old = json.loads((OLD / "panels/stage0.json").read_text())
ordered = list(old["validation"]["20"])
for key in ["60", "180"]:
    ordered += [b for b in old["validation"][key] if b not in set(ordered)]
extra = sorted(
    set(validation.index) - set(ordered),
    key=lambda b: hashlib.sha256(("mid-panel0\0" + b).encode()).hexdigest(),
)
ordered += extra
panels = {
    "validation": {
        "1": old["validation"]["1"],
        "20": ordered[:400],
        "60": ordered[:1200],
        "180": ordered[:3548],
    },
    "test": old["test"],
}
for key, n in [("20", 400), ("60", 1200), ("180", 3548)]:
    ids = panels["validation"][key]
    assert len(ids) == n and len(set(ids)) == n
    assert comparison_components(validation.loc[ids].reset_index(), models) == [
        frozenset(models)
    ]
    assert not set(ids) & set(old["test"]["100"])
manifest["panel_sha256"] = hashlib.sha256(
    json.dumps(panels, sort_keys=True).encode()
).hexdigest()
(LOCAL / "panels/panels.json").write_text(json.dumps(panels, indent=2))
base = yaml.safe_load((OLD / "configs/calibration-base.yaml").read_text())
base["meta_eval"]["panel_file"] = str(REMOTE / "panels/panels.json")
base["run"]["result_folder"] = str(REMOTE / "results")
(LOCAL / "configs/calibration-base.yaml").write_text(
    yaml.safe_dump(base, sort_keys=False)
)
(LOCAL / "configs/manifest.json").write_text(json.dumps(manifest, indent=2))
for arm in ["uniform", "primo-none", "primo-prior"]:
    cfg = json.loads(json.dumps(base))
    cfg["task"] = "tune-judge"
    centers = (
        {"agreement": manifest["agreement_prior"], "cost": manifest["cost_prior"]}
        if arm == "primo-prior"
        else {}
    )
    optimizer = (
        {
            "name": "mo_hyperband",
            "sampler": "uniform",
            "eta": 3,
            "mo_selector": "epsnet",
        }
        if arm == "uniform"
        else {
            "name": "primo",
            "sampler": "uniform",
            "eta": 3,
            "mo_selector": "epsnet",
            "initial_design_size": 2,
            "device": "cpu",
            "prior_centers": centers,
            "prior_confidences": {o: {p: 0.5 for p in c} for o, c in centers.items()},
        }
    )
    cfg["tune_judge"] = {
        "meta_eval_task": "meta-eval-lmarena-140k-en",
        "objectives": ["agreement", "cost"],
        "objective_bounds": manifest["objective_bounds"],
        "fidelity": {"battles_per_model": {"lower": 20, "upper": 180}},
        "search_space": {
            "judge": {
                "model": manifest["models"],
                "prompt_preset": manifest["presets"],
                "swap_mode": ["fixed", "both"],
            }
        },
        "optimizer": optimizer,
        "neps": {"total_fidelities_to_spend": 1440, "ignore_errors": True},
        "price_per_million_tokens": manifest["prices"],
        "skip_test": True,
        "seed": 0,
        "run_dir": str(REMOTE / "searches" / arm),
    }
    RunConfig(**cfg)
    (LOCAL / "configs" / f"{arm}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("FROZEN60", len(meta), "PANELS40012003548", flush=True)
