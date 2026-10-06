"""Local commands for ``judgearena leaderboard``.

Create a benchmark, evaluate candidates, show results, export files, or validate
saved submissions. Ordinary Elo runs enter through ``judgearena.cli`` instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from judgearena.benchmarks.elo.artifacts import (
    export_leaderboard,
    validate_submission,
)
from judgearena.benchmarks.elo.freeze import freeze_leaderboard
from judgearena.benchmarks.runner import run_benchmark
from judgearena.config import RunConfig, _resolve_prompt_paths, load_config
from judgearena.log import configure_logging


class LeaderboardSetup(BaseModel):
    """Creation settings plus the ordinary JudgeArena evaluation config."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    languages: list[str] = Field(min_length=1)
    anchor_models: list[str] = Field(min_length=1)
    battles_per_language: int = Field(gt=0)
    min_anchor_battles: int = Field(gt=0)
    evaluation: RunConfig


def load_setup(path: Path) -> LeaderboardSetup:
    """Read creation settings and the evaluation config from a YAML setup file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("A leaderboard setup must contain a YAML mapping.")
    if isinstance(data.get("evaluation"), dict):
        _resolve_prompt_paths(data["evaluation"], path)
    return LeaderboardSetup.model_validate(data)


def show_leaderboard(directory: Path) -> None:
    """Print saved reference and candidate ratings without running evaluation."""
    board = json.loads((directory / "leaderboard.json").read_text())
    rows = []
    for entry in board["entries"]:
        overall = entry["overall"]
        interval = (
            f"{overall['ci_low']:.0f}–{overall['ci_high']:.0f}"
            if overall["ci_low"] is not None
            else "—"
        )
        rows.append(
            {
                "Model": entry["model"],
                "Source": "Human anchor"
                if entry["source"] == "anchor"
                else "Judge estimate",
                "Elo": round(overall["rating"], 1),
                "95% CI": interval,
                "Battles": overall["n_battles"],
                **{
                    language: round(summary["rating"], 1)
                    for language, summary in entry["by_language"].items()
                },
            }
        )
    print(f"{board['name']} v{board['version']}")
    print(pd.DataFrame(rows).to_string(index=False))


def run_leaderboard_command(argv: list[str]) -> None:
    """Handle ``judgearena leaderboard`` commands; all writes are local."""
    parser = argparse.ArgumentParser(prog="judgearena leaderboard")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Freeze a version from a YAML setup.")
    create.add_argument("setup", type=Path)
    create.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser(
        "evaluate",
        aliases=["submit"],
        help="Evaluate one model and save results locally.",
    )
    evaluate.add_argument("directory", type=Path)
    evaluate.add_argument("--model", required=True)
    show = commands.add_parser("show", help="Show overall and per-language ratings.")
    show.add_argument("directory", type=Path)
    export = commands.add_parser(
        "export", help="Export a portable dataset tree locally."
    )
    export.add_argument("directory", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--results-dir", type=Path)
    validate = commands.add_parser(
        "validate-submission",
        help="Recompute an entry from saved battles; no inference.",
    )
    validate.add_argument("directory", type=Path)
    validate.add_argument("--entry", type=Path, required=True)
    validate.add_argument("--battles", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "create":
            setup = load_setup(args.setup)
            configure_logging(setup.evaluation.run.verbosity)
            output = freeze_leaderboard(
                setup.evaluation,
                args.output,
                setup.anchor_models,
                setup.languages,
                setup.battles_per_language,
                name=setup.name,
                version=setup.version,
                min_anchor_battles=setup.min_anchor_battles,
            )
            print(f"Created {setup.name} v{setup.version} at {output}")
        elif args.command in ("evaluate", "submit"):
            cfg = load_config(args.directory / "config.yaml")
            cfg.model.name = args.model
            if cfg.elo is None:
                raise ValueError("A leaderboard requires an Elo configuration.")
            cfg.elo.leaderboard_dir = args.directory
            configure_logging(cfg.run.verbosity, log_file=cfg.run.log_file)
            result = run_benchmark(cfg)
            print(f"Saved run: {Path(result['result_path']).parent}")
        elif args.command == "export":
            output = export_leaderboard(
                args.directory, args.output, results_dir=args.results_dir
            )
            print(f"Exported to {output}")
        elif args.command == "validate-submission":
            entry = validate_submission(args.directory, args.entry, args.battles)
            print(f"Validated {entry.model}: {entry.overall.rating:.1f} Elo")
        else:
            show_leaderboard(args.directory)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
