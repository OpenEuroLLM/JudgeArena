"""Smoke checks for packaged wheel/sdist artifacts.

This file is executed directly in isolated environments by the publish workflow.
"""

from __future__ import annotations

from importlib.metadata import version
from importlib.resources import files

from judgearena.criteria.defaults import CRITERIA_BY_NAME

META_EVAL_PROMPT_RESOURCES = (
    "arena_hard_system.txt",
    "arena_hard_user.txt",
    "alpaca_eval_system.txt",
    "alpaca_eval_user.txt",
    "alpaca_eval_pair_score_user.txt",
)


def _assert_non_empty_text_resource(package: str, relative_path: str) -> None:
    content = files(package).joinpath(relative_path).read_text()
    if not content.strip():
        raise AssertionError(f"Expected non-empty resource: {package}/{relative_path}")


def main() -> None:
    print(f"Starting smoke test for JudgeArena {version('judgearena')}...")

    # Validates that package import and built-in criteria loading both work.
    if "default" not in CRITERIA_BY_NAME:
        raise AssertionError("Missing default criteria entry.")
    if not CRITERIA_BY_NAME["default"]:
        raise AssertionError("Default criteria list is empty.")

    # Validates packaged text resources expected at runtime.
    _assert_non_empty_text_resource("judgearena.prompts", "prompt.txt")
    _assert_non_empty_text_resource("judgearena.prompts", "system-prompt.txt")
    _assert_non_empty_text_resource("judgearena.criteria", "data/default.yaml")
    for filename in META_EVAL_PROMPT_RESOURCES:
        _assert_non_empty_text_resource(
            "judgearena.meta_eval",
            f"prompts/{filename}",
        )

    print("✅ All integrity checks passed: Imports, Criteria, and Resources are valid.")


if __name__ == "__main__":
    main()
