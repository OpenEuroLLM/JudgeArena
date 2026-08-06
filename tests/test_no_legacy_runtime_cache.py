"""Regression guard against reintroducing legacy runtime caches."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGEARENA_ROOT = REPO_ROOT / "judgearena"

FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AnnotationCache", re.compile(r"\bAnnotationCache\b")),
    ("cache_function_dataframe", re.compile(r"\bcache_function_dataframe\b")),
    ("generation_cache_token", re.compile(r"\bgeneration_cache_token\b")),
    ("ignore_cache", re.compile(r"\bignore_cache\b")),
    ("set_langchain_cache", re.compile(r"\bset_langchain_cache\b")),
    ("legacy cache/db path", re.compile(r"cache/db")),
]

FORBIDDEN_IMPORTS = (
    "from judgearena.utils import cache_function_dataframe",
    "from judgearena.utils.io import cache_function_dataframe",
    "from judgearena.meta_eval.cache import",
)

INFERENCE_BATCH_INVOKE_ALLOWLIST = frozenset(
    {
        "judgearena/models.py",
        "judgearena/model_adapters.py",
        "judgearena/generate.py",
        "judgearena/evaluate.py",
        "judgearena/mt_bench/pairwise_judging.py",
    }
)


def _judgearena_py_files() -> list[Path]:
    return sorted(JUDGEARENA_ROOT.rglob("*.py"))


def test_meta_eval_cache_module_is_absent() -> None:
    assert not (JUDGEARENA_ROOT / "meta_eval" / "cache.py").exists()


def test_judgearena_has_no_legacy_runtime_cache_symbols() -> None:
    violations: list[str] = []
    for path in _judgearena_py_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                violations.append(f"{rel}: {label}")
        for imp in FORBIDDEN_IMPORTS:
            if imp in text:
                violations.append(f"{rel}: import {imp}")
    assert not violations, "Legacy runtime cache references found:\n" + "\n".join(
        violations
    )


def test_model_inference_batch_invoke_is_allowlisted() -> None:
    pattern = re.compile(r"\.(batch|invoke)\(")
    violations: list[str] = []
    for path in _judgearena_py_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in INFERENCE_BATCH_INVOKE_ALLOWLIST:
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            violations.append(rel)
    assert not violations, (
        "Direct .batch/.invoke outside allowlist (route through do_inference):\n"
        + "\n".join(violations)
    )
