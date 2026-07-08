"""ipywidgets browser for JudgeArena annotations.

Run from the terminal:
    uv run python judgearena/browse_widgets.py [--store-root ~/judgearena-data/db]

This opens a Jupyter Notebook in your browser with an interactive UI.
Inside the notebook the widget is also importable directly:
    from judgearena.browse_widgets import make_ui; make_ui()
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Widget UI — called once per Jupyter session
# ---------------------------------------------------------------------------


def make_ui(
    store_root: Path | str | None = None, csv_dir: Path | str | None = None
) -> None:
    """Display the annotation browser widget (must be called inside Jupyter).

    Args:
        store_root: Root of the SQLite store (default: ~/judgearena-data/db).
        csv_dir: Directory containing the original judge CSV / CSV.zip files.
                 When provided, instruction text and completions are loaded from
                 those files (always correct) rather than the live HF arena DF
                 (which can be misaligned if the dataset grew after generation).
    """
    import ipywidgets as w
    from IPython.display import Markdown, clear_output, display

    from judgearena.browse_cache import (
        STORE_ROOT as _DEFAULT_ROOT,
    )
    from judgearena.browse_cache import (
        list_judges,
        list_languages,
        list_models,
        load_context,
        load_subset,
        load_subset_from_csv,
    )

    root = Path(store_root).expanduser() if store_root else _DEFAULT_ROOT

    # Auto-detect csv_dir from common project locations if not specified
    if csv_dir is not None:
        _csv_dir = Path(csv_dir).expanduser()
    else:
        _candidates = [
            Path(__file__).parent.parent
            / "slurmpilot_scripts"
            / "olmo3-evals"
            / "data",
            Path(__file__).parent.parent / "slurmpilot_scripts" / "elaine_evals",
        ]
        _csv_dir = next((p for p in _candidates if p.is_dir()), None)

    if _csv_dir is not None:
        print(f"Using judge CSVs from {_csv_dir} for correct instruction alignment.")
    else:
        print(
            "Loading arena context (disk cache, ~4 s first time)…", end=" ", flush=True
        )
        load_context()
        print("done.")

    models = list_models(root)
    judges = list_judges(root)
    languages = list_languages()

    if not models:
        print(f"No completions found under {root}")
        return

    # --- state ---
    _state: dict = {"df": pd.DataFrame(), "idx": 0}

    # --- widgets ---
    model_dd = w.Dropdown(
        options=models,
        description="Model:",
        style={"description_width": "80px"},
        layout=w.Layout(width="420px"),
    )
    lang_dd = w.Dropdown(
        options=["All"] + languages,
        value="All",
        description="Language:",
        style={"description_width": "80px"},
        layout=w.Layout(width="200px"),
    )
    judge_dd = w.Dropdown(
        options=["All"] + judges,
        value="All",
        description="Judge:",
        style={"description_width": "80px"},
        layout=w.Layout(width="420px"),
    )
    prev_btn = w.Button(description="◀", layout=w.Layout(width="40px"))
    next_btn = w.Button(description="▶", layout=w.Layout(width="40px"))
    nav_label = w.Label("—", layout=w.Layout(width="80px"))
    out_header = w.Output()
    out_a = w.Output(layout=w.Layout(width="50%", padding="0 8px 0 0"))
    out_b = w.Output(layout=w.Layout(width="50%", padding="0 0 0 8px"))
    out_verdict = w.Output()

    def _current_filters():
        lang = None if lang_dd.value == "All" else lang_dd.value
        judge = None if judge_dd.value == "All" else judge_dd.value
        return model_dd.value, lang, judge

    def _refresh_nav():
        total = len(_state["df"])
        idx = _state["idx"]
        nav_label.value = f"{idx + 1}/{total}" if total else "—"
        prev_btn.disabled = idx <= 0 or total == 0
        next_btn.disabled = idx >= total - 1

    def _pref(p) -> str:
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return "*(could not parse)*"
        if p < 0.45:
            return f"**A wins** ({p:.2f})"
        if p > 0.55:
            return f"**B wins** ({p:.2f})"
        return f"Tie ({p:.2f})"

    def _show():
        df = _state["df"]

        with out_header:
            clear_output(wait=True)
            if df.empty:
                display(Markdown("*No data for the selected filters.*"))
                return
            row = df.iloc[_state["idx"]]
            instruction = (row.get("instruction") or "").strip()
            idx_tag = ""
            if "instruction_index" in df.columns:
                src = f"  `{row['source']}`" if "source" in df.columns else ""
                idx_tag = f"  *(idx={int(row['instruction_index'])}{src})*"
            display(
                Markdown(
                    f"**Model A:** `{row['model_a']}`  **Model B:** `{row['model_b']}`{idx_tag}\n\n"
                    f"---\n\n### Instruction\n\n{instruction or '*(empty)*'}"
                )
            )

        if df.empty:
            with out_a:
                clear_output(wait=True)
            with out_b:
                clear_output(wait=True)
            with out_verdict:
                clear_output(wait=True)
            return

        row = df.iloc[_state["idx"]]
        comp_a = (row.get("completion_a") or "").strip()
        comp_b = (row.get("completion_b") or "").strip()
        judgement = (row.get("judgement") or "").strip()

        with out_a:
            clear_output(wait=True)
            display(
                Markdown(
                    f"### Completion A\n*{row['model_a']}*\n\n"
                    f"{comp_a or '*(not found in store)*'}"
                )
            )

        with out_b:
            clear_output(wait=True)
            display(
                Markdown(
                    f"### Completion B\n*{row['model_b']}*\n\n"
                    f"{comp_b or '*(not found in store)*'}"
                )
            )

        with out_verdict:
            clear_output(wait=True)
            display(
                Markdown(
                    f"---\n\n### Judge verdict\n\n{_pref(row.get('preference'))}\n\n{judgement}"
                )
            )

    def _reload(_change=None):
        model, lang, judge = _current_filters()
        languages_filter = [lang] if lang else None
        with out_header:
            clear_output(wait=True)
            display(Markdown("*Loading…*"))
        try:
            if _csv_dir is not None:
                df = load_subset_from_csv(
                    model=model,
                    csv_dir=_csv_dir,
                    languages=languages_filter,
                )
            else:
                df = load_subset(
                    model=model,
                    languages=languages_filter,
                    judge=judge,
                    store_root=root,
                )
        except Exception as exc:
            with out_header:
                clear_output(wait=True)
                display(Markdown(f"**Error:** `{exc}`"))
            return
        _state["df"] = df
        _state["idx"] = 0
        _refresh_nav()
        _show()

    def _go(delta: int):
        df = _state["df"]
        if df.empty:
            return
        _state["idx"] = max(0, min(len(df) - 1, _state["idx"] + delta))
        _refresh_nav()
        _show()

    model_dd.observe(_reload, names="value")
    lang_dd.observe(_reload, names="value")
    judge_dd.observe(_reload, names="value")
    prev_btn.on_click(lambda _: _go(-1))
    next_btn.on_click(lambda _: _go(1))

    # Trigger an initial load
    _reload()

    controls = w.VBox(
        [
            model_dd,
            w.HBox([lang_dd, judge_dd]),
            w.HBox([prev_btn, nav_label, next_btn]),
        ]
    )
    display(
        w.VBox(
            [
                controls,
                out_header,
                w.HBox([out_a, out_b]),
                out_verdict,
            ]
        )
    )


# ---------------------------------------------------------------------------
# CLI: self-launch inside a Jupyter Notebook
# ---------------------------------------------------------------------------


def _launch() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store-root",
        type=Path,
        default=Path.home() / "judgearena-data" / "db",
        help="Root of the SQLite store (default: ~/judgearena-data/db)",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Directory with original judge CSV/CSV.zip files for correct instruction alignment.",
    )
    args = parser.parse_args()
    store_root = str(args.store_root.expanduser())
    csv_dir = str(args.csv_dir.expanduser()) if args.csv_dir else None

    csv_arg = f", csv_dir={csv_dir!r}" if csv_dir else ""

    # Build a one-cell notebook that calls make_ui()
    src = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(Path(__file__).parent.parent)!r})",
            "from judgearena.browse_widgets import make_ui",
            f"make_ui(store_root={store_root!r}{csv_arg})",
        ]
    )
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "code",
                "id": "browse-main",
                "source": src,
                "outputs": [],
                "metadata": {},
                "execution_count": None,
            }
        ],
    }

    with tempfile.NamedTemporaryFile(
        suffix=".ipynb", delete=False, mode="w", prefix="judgearena_browse_"
    ) as f:
        json.dump(nb, f)
        tmp = f.name

    print(f"Opening browser… (notebook at {tmp})")
    try:
        subprocess.run(
            [sys.executable, "-m", "jupyter", "notebook", tmp],
            check=True,
        )
    finally:
        Path(tmp).unlink(missing_ok=True)


if __name__ == "__main__":
    _launch()
