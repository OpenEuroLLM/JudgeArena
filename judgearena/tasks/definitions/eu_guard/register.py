"""Register EU-Guard data, metrics and reusable judge bundles."""

from pathlib import Path

from judgearena.benchmarks.scoring import register_metric
from judgearena.datasets.registry import register_dataset
from judgearena.prompts import JudgePromptPreset, register_prompt
from judgearena.tasks.definitions.eu_guard.dataset import EUGuardAdapter
from judgearena.tasks.definitions.eu_guard.metrics import EUGuardMetric

register_dataset("eu_guard")(EUGuardAdapter)
register_metric("eu_guard")(EUGuardMetric)

for name in ("guard-jev", "guard-gemma", "refusal-jev"):
    register_prompt(
        JudgePromptPreset.from_file(
            f"eu-guard-{name}", Path(__file__).parent / "prompts" / f"{name}.yaml"
        )
    )
