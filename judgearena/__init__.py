"""Public entrypoints, loaded on first use to keep lightweight imports cheap."""

from importlib import import_module

_EXPORTS = {
    "RunConfig": "judgearena.config",
    "run_benchmark": "judgearena.benchmarks.runner",
    "CompletionData": "judgearena.datasets.data",
    "PreferenceData": "judgearena.datasets.data",
    "load_tasks": "judgearena.tasks.registry",
    "DatasetAdapter": "judgearena.datasets.registry",
    "register_dataset": "judgearena.datasets.registry",
    "register_prompt": "judgearena.prompts",
    "register_metric": "judgearena.benchmarks.scoring",
    "register_parser": "judgearena.prompts",
    "JudgePromptPreset": "judgearena.prompts",
    "JudgeParser": "judgearena.prompts",
    "OutputParser": "judgearena.prompts",
    "ParsedPreference": "judgearena.prompts",
    "ParsedScore": "judgearena.prompts",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
