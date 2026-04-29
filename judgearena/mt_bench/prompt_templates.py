from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "mt_bench"
_USER_SINGLE_BASE_FILE = "user-single-base.txt"
_USER_MULTI_BASE_FILE = "user-multi-base.txt"
_USER_SINGLE_REF_BLOCK_FILE = "user-single-reference-block.txt"
_USER_MULTI_REF_BLOCK_FILE = "user-multi-reference-block.txt"


def load_mt_bench_prompt_text(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


def render_mt_bench_prompt_text(filename: str, **kwargs: str) -> str:
    return load_mt_bench_prompt_text(filename).format(**kwargs)


def build_mt_bench_user_prompt_template(*, multi_turn: bool, ref_based: bool) -> str:
    base_filename = _USER_MULTI_BASE_FILE if multi_turn else _USER_SINGLE_BASE_FILE
    reference_block = ""
    if ref_based:
        ref_block_filename = (
            _USER_MULTI_REF_BLOCK_FILE if multi_turn else _USER_SINGLE_REF_BLOCK_FILE
        )
        reference_block = (
            load_mt_bench_prompt_text(ref_block_filename).rstrip("\n") + "\n\n"
        )
    return render_mt_bench_prompt_text(base_filename, reference_block=reference_block)
