"""Relocate frozen experiment inputs, run workers and regenerate analyses."""

import argparse
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL_NAMES = {
    "intermediate": "judge-tuning-intermediate-20261014",
    "heldout": "judge-tuning-heldout-3000-20261015",
}


def relocate(work):
    replacements = {}
    for stage, name in ORIGINAL_NAMES.items():
        for prefix in [
            "/private/tmp/judgearena-tuning-",
            "/work/dlclarge1/lushtake-hiwi/judge-tuning-",
        ]:
            suffix = name.removeprefix("judge-tuning-")
            replacements[prefix + suffix] = str(work / stage)
    replacements["/work/dlclarge1/lushtake-hiwi/judge-tuning-local-20261012/cache"] = (
        str(work / "cache")
    )
    replacements["/private/tmp/judgearena-tuning-v2"] = str(work / "preparation-inputs")
    for stage in ["intermediate", "heldout", "preparation-inputs"]:
        source = HERE / "archive" / stage
        shutil.copytree(source, work / stage, dirs_exist_ok=True)
    for path in work.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yaml", ".json", ".sh"}:
            continue
        text = path.read_text()
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text)
    link = work / "intermediate/source"
    if not link.exists():
        link.symlink_to(REPO, target_is_directory=True)
    manifest_path = work / "heldout/configs/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for candidate in manifest["candidates"]:
        for field, digest in [
            ("config_file", "sha256"),
            ("source_config", "source_sha256"),
        ]:
            candidate[digest] = hashlib.sha256(
                Path(candidate[field]).read_bytes()
            ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("Prepared", work, "(archived originals remain unchanged in the repository)")


def execute(work, stage, script, arguments=()):
    sys.path.insert(0, str(REPO))
    sys.argv = [script, *arguments]
    runpy.run_path(str(work / stage / "launch" / script), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "analyze", "search", "test"])
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--arm", choices=["uniform", "primo-none", "primo-prior"])
    parser.add_argument("--group", choices=["gemma", "qwen"])
    args = parser.parse_args()
    work = args.workdir.resolve()
    if args.action == "prepare":
        if work.exists():
            parser.error(
                "Use a new workdir so previous runs and provenance remain intact"
            )
        relocate(work)
        return
    if not (work / "heldout/configs/manifest.json").exists():
        parser.error("Run prepare first")
    if args.action == "analyze":
        for script in ["analyze.py", "render_report.py", "plot_trajectories.py"]:
            execute(work, "intermediate", script)
        execute(work, "heldout", "analyze_test.py")
    elif args.action == "search":
        if args.arm is None:
            parser.error("search requires --arm")
        # Each arm gets a fresh NePS root; existing trials are resumed by native NePS.
        execute(work, "intermediate", "search.py", [args.arm])
    else:
        if args.group is None:
            parser.error("test requires --group")
        execute(work, "heldout", "evaluate_test.py", [args.group])


if __name__ == "__main__":
    main()
