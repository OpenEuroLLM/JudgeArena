"""Convert a saved config to the current schema beside the original file."""

import argparse
import re
from copy import deepcopy
from pathlib import Path

import yaml

from judgearena.deprecations import translate_deprecations
from judgearena.prompts.spec import config_prompts
from judgearena.tasks.registry import _canonical_sha256


def migrate_saved_template_fields(prompt, directory):
    """Rename placeholders in a loaded snapshot without rewriting its archived files."""
    from judgearena.prompts.registry import file_prompt_name

    original_user = prompt.get("user_file")
    original_system = prompt.get("system_file")
    for field in ("user", "system"):
        file_field = f"{field}_file"
        path = prompt.get(file_field)
        text = (
            (directory / path).read_text(encoding="utf-8")
            if path
            else prompt.get(field)
        )
        if text is None:
            continue

        def rename(match):
            key = match[1]
            if key is None:  # Escaped opening braces introduce literal text.
                return match[0]
            return "{" + key.replace("user_prompt", "instruction").lower()

        migrated = re.sub(
            r"\{\{|\{((?:user_prompt|completion_[AB](?:_text)?)(?:_json)?)(?=[}:!])",
            rename,
            text,
        )
        if migrated != text:
            if path and original_user and not prompt.get("name"):
                prompt["name"] = file_prompt_name(original_system, original_user)
            prompt.pop(file_field, None)
            prompt[field] = migrated


def _finish_snapshot(data):
    # Saved run options override protocol defaults; the resolved task also needs
    # the new opponent policy for loading outside the original task registry.
    resolved = data["resolved_task"]
    protocol = resolved["spec"]["protocol"]
    task_graph = protocol.pop("graph", None)
    if task_graph is not None:
        protocol["baseline"] = {"strategy": "recorded_opponents"}
        if (
            data.get("graph") is None
            and (data.get("protocol") or {}).get("graph") is None
        ):
            data["graph"] = task_graph
    data = translate_deprecations(data, protocol)
    spec = data["resolved_task"]["spec"]
    data["resolved_task"]["provenance"]["resolved_sha256"] = _canonical_sha256(spec)
    return data


def migrate_snapshot(data: dict, directory: Path) -> dict:
    """Read old run snapshots without exposing their options to new experiments."""
    data = deepcopy(data)
    resolved = data.get("resolved_task") if isinstance(data, dict) else None
    if not isinstance(resolved, dict):
        raise ValueError("Expected a saved run containing resolved_task.")
    for prompt in config_prompts(data):
        migrate_saved_template_fields(prompt, directory)
    spec = resolved["spec"]
    protocol = spec["protocol"]
    saved_format = spec["dataset"].get("format")
    for dataset in (spec["dataset"], data.get("dataset") or {}):
        if not ({"format", "battle_columns"} & dataset.keys()) and dataset.get(
            "adapter"
        ) not in {"arena_battles", "judgearena_tables"}:
            continue
        release = dataset.pop("format", None)
        if dataset.pop("battle_columns", {}) or release == "battles":
            raise ValueError(
                "Saved battle-table configs require a custom dataset adapter."
            )
        if dataset.get("adapter") == "judgearena_tables":
            dataset["adapter"] = "alpaca_eval"
        if dataset.get("adapter") == "arena_battles":
            release = release or saved_format or protocol.get("arena")
            if release == "ComparIA":
                dataset["adapter"] = "comparia"
            elif release in {"LMArena", "LMArena-55k", "LMArena-100k", "LMArena-140k"}:
                dataset["adapter"] = "lmarena"
            else:
                raise ValueError(f"Unknown saved arena format {release!r}")
        # A saved release selector could narrow a combined source list.
        if dataset.get("adapter") == "lmarena" and release and release != "LMArena":
            repo_id = (
                f"lmarena-ai/arena-human-preference-{release.removeprefix('LMArena-')}"
            )
            dataset["sources"] = {
                key: source
                for key, source in dataset["sources"].items()
                if source.get("repo_id") == repo_id
            }
            if not dataset["sources"]:
                raise ValueError(
                    f"No declared dataset source matches saved format {release!r}."
                )
    for graph in (
        data.get("graph"),
        (data.get("protocol") or {}).get("graph"),
        protocol.get("graph"),
    ):
        if isinstance(graph, dict) and "per_language" in graph:
            quota = graph.pop("per_language")
            if quota is not None:
                graph.setdefault("group_by", "lang")
                graph.setdefault("per_group", quota)
    options = data.pop("elo", None) or {}
    old_runner = protocol["runner"] == "elo"
    if not old_runner and not options:
        return _finish_snapshot(data)
    if protocol["runner"] not in {"elo", "pairwise"}:
        raise ValueError("Saved Elo settings require a pairwise task")

    scoring = protocol["scoring"]
    metrics = scoring["metrics"]
    defaults = next(
        (
            m.get("parameters", {})
            for m in metrics
            if m["metric"] == "bradley_terry"
            and m.get("name") in (None, "bradley_terry")
        ),
        {},
    )
    soft = options.get(
        "soft_elo", defaults.get("soft", scoring.get("default_soft", True))
    )
    temperature = options.get(
        "soft_elo_temperature", scoring.get("default_temperature", 0.3)
    )
    # Before frozen fitting existed, omitted fit_mode meant a joint fit.
    fit_mode = options.get("fit_mode", defaults.get("fit_mode", "joint"))
    graph = data.get("graph") or protocol.get("graph") or {}
    for old, new in (
        ("elo_random_battles", "random_battles"),
        ("n_instructions_per_language", "per_group"),
    ):
        if options.get(old) is not None:
            graph.setdefault(new, options[old])
    if graph.get("per_group") is not None:
        graph.setdefault("group_by", "lang")
    protocol.update(runner="pairwise", graph=graph)
    data["graph"] = graph
    protocol.pop("arena", None)
    judge = protocol["judge"]
    for key, value in {
        "preference_temperature": temperature if soft else None,
        "calibrate_temperature": options.get("calibrate_temperature", False) and soft,
        "calibration_size": options.get("calibration_size"),
    }.items():
        if judge.get(key) is None:
            judge[key] = value
    for metric in metrics:
        if metric["metric"] != "bradley_terry":
            continue
        parameters = metric.setdefault("parameters", {})
        migrated = {
            "fit_mode": fit_mode,
            "soft": soft,
            "baseline_model": options.get(
                "baseline_model", defaults.get("baseline_model")
            ),
            "n_bootstraps": options.get(
                "n_bootstraps", defaults.get("n_bootstraps", 20)
            ),
        }
        # The old runner applied its runtime overrides to the default metric only.
        if metric.get("name") in (None, "bradley_terry"):
            parameters.update(migrated)
        else:
            for key, value in migrated.items():
                parameters.setdefault(key, value)
        parameters.setdefault(
            "reference_min_battles", 500 if parameters["fit_mode"] == "joint" else 0
        )
    scoring.pop("default_soft", None)
    scoring.pop("default_temperature", None)
    if options.get("languages"):
        from judgearena.datasets import resolve_dataset_filters

        run = data.setdefault("run", {})
        run["filters"] = resolve_dataset_filters(
            None, run.get("filters", {}), {"lang": options["languages"]}
        )
    data.update(protocol=protocol, dataset=spec["dataset"])
    return _finish_snapshot(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    path = parser.parse_args().config
    migrated = migrate_snapshot(
        yaml.safe_load(path.read_text(encoding="utf-8")), path.parent
    )
    output = path.with_stem(path.stem + ".migrated")
    with output.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(migrated, stream, sort_keys=False)
    print(output)
