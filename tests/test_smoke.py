"""Smoke checks for packaged wheel/sdist artifacts.

This file is executed directly in isolated environments by the publish workflow.
"""

from __future__ import annotations

from importlib.metadata import version
from importlib.resources import files

from judgearena.prompts.criteria import resolve_criteria
from judgearena.prompts.registry import resolve_judge_prompt, resolve_prompt_spec
from judgearena.prompts.spec import JudgePromptSpec
from judgearena.tasks import load_tasks


def _assert_non_empty_text_resource(package: str, relative_path: str) -> None:
    content = files(package).joinpath(relative_path).read_text()
    if not content.strip():
        raise AssertionError(f"Expected non-empty resource: {package}/{relative_path}")


def main() -> None:
    print(f"Starting smoke test for JudgeArena {version('judgearena')}...")

    # Validates that package import and built-in criteria loading both work.
    if not resolve_criteria():
        raise AssertionError("Default criteria list is empty.")

    # Validates packaged text resources expected at runtime.
    _assert_non_empty_text_resource("judgearena.prompts", "templates/prompt.txt")
    _assert_non_empty_text_resource("judgearena.prompts", "templates/system-prompt.txt")
    _assert_non_empty_text_resource(
        "judgearena.prompts", "templates/criteria-default.yaml"
    )

    tasks = load_tasks()
    for name in ("arena-hard-v0.1", "alpaca-eval", "mt-bench", "wildbench-score"):
        if name not in tasks:
            raise AssertionError(f"Packaged task not discovered: {name}")

    bundle = files("judgearena.tasks").joinpath(
        "definitions/arena_hard/prompts/pairwise.yaml"
    )
    prompt = resolve_prompt_spec(JudgePromptSpec(file=str(bundle)))
    registered = resolve_judge_prompt(preset="arena-hard")
    if (
        not prompt.system_prompt
        or not prompt.user_prompt_template
        or prompt.user_prompt_template != registered.user_prompt_template
        or prompt.judging_mode != "pairwise"
        or prompt.parser.parse_result("[[A>B]]").preference != 0.25
    ):
        raise AssertionError("Packaged prompt bundle or task-local parser is invalid.")

    print(f"Passed: shared resources, {len(tasks)} tasks, prompt bundle and parser.")


if __name__ == "__main__":
    main()
