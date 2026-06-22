"""Gradio leaderboard Space (thin renderer over a published bundle)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# When deployed, app.py and render.py sit side-by-side at the Space root
# (`from render import ...`); in this repo they live under `space/` as a
# namespace package (`from space.render import ...`). gradio is imported lazily
# (inside build_demo) so importing this module for tests does not require it.
try:
    from render import (
        available_languages,
        calibration_fig,
        distribution_fig,
        head_to_head_heatmap,
        header_html,
        kappa_bar,
        language_bar,
        language_table,
        overview_table_html,
        winrate_heatmap,
    )
except ModuleNotFoundError:
    from space.render import (
        available_languages,
        calibration_fig,
        distribution_fig,
        head_to_head_heatmap,
        header_html,
        kappa_bar,
        language_bar,
        language_table,
        overview_table_html,
        winrate_heatmap,
    )

_DATASET_REPO = "OpenEuroLLM/judgearena-leaderboard"  # default published dataset


def _resolve_local_version(base: "Path") -> str:
    versions = sorted(
        p.name for p in (base / "panel").iterdir() if p.is_dir()
    )
    if not versions:
        raise SystemExit("space: no panel/* under the local dir.")
    return versions[-1]


def load_bundle(
    local_dir: str | None = None, repo: str | None = None, panel_version: str | None = None
) -> tuple[dict, pd.DataFrame]:
    """Assemble (bundle, scores) from a local dir or a HF dataset repo."""
    try:
        from assemble import assemble_from_dirs
    except ModuleNotFoundError:
        from judgearena.leaderboard.assemble import assemble_from_dirs

    if local_dir:
        base = Path(local_dir)
        version = panel_version or _resolve_local_version(base)
        return assemble_from_dirs(base / "panel" / version, base / "records" / version)

    from huggingface_hub import HfApi, snapshot_download

    repo = repo or _DATASET_REPO
    version = panel_version
    if version is None:
        files = HfApi().list_repo_files(repo_id=repo, repo_type="dataset")
        versions = sorted(
            {f.split("/")[1] for f in files if f.startswith("panel/") and "/" in f[6:]}
        )
        if not versions:
            raise SystemExit("space: no panel/* in the dataset repo.")
        version = versions[-1]
    local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=[f"panel/{version}/*.json", f"records/{version}/**"],
    )
    base = Path(local)
    return assemble_from_dirs(base / "panel" / version, base / "records" / version)


def build_demo(bundle: dict, scores: pd.DataFrame):  # -> gr.Blocks
    import gradio as gr

    langs = available_languages(bundle)
    models = sorted(scores["model"].unique().tolist()) if len(scores) else []
    theme = gr.themes.Soft(font=gr.themes.GoogleFont("Inter"))

    with gr.Blocks(theme=theme, title="JudgeArena Leaderboard") as demo:
        gr.HTML(header_html(bundle))
        with gr.Tab("Overview"):
            gr.Plot(kappa_bar(bundle))
            lang_choices = ["All", *langs]
            lang_dd = gr.Dropdown(lang_choices, value="All", label="Language")
            table = gr.HTML(overview_table_html(bundle, None))

            def _on_lang(choice):
                return overview_table_html(bundle, None if choice == "All" else choice)

            lang_dd.change(_on_lang, lang_dd, table)
        with gr.Tab("Calibration"):
            gr.Plot(calibration_fig(bundle))
        with gr.Tab("Win rates"):
            gr.Plot(winrate_heatmap(bundle))
            if langs:
                wl = gr.Dropdown(langs, value=langs[0], label="Language")
                wt = gr.Dataframe(value=language_table(bundle, langs[0]), interactive=False)
                wb = gr.Plot(language_bar(bundle, langs[0]))
                wl.change(lambda lg: (language_table(bundle, lg), language_bar(bundle, lg)),
                          wl, [wt, wb])
        with gr.Tab("Head-to-head"):
            gr.Plot(head_to_head_heatmap(bundle))
        with gr.Tab("Distributions"):
            if models:
                default = models[: min(3, len(models))]
                ms = gr.Dropdown(models, value=default, multiselect=True, label="Models")
                dp = gr.Plot(distribution_fig(scores, default))
                ms.change(lambda sel: distribution_fig(scores, sel), ms, dp)
            else:
                gr.Markdown("_No per-battle scores published yet._")
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="judgearena-space")
    ap.add_argument("--local", default=None, help="Local dataset dir (panel/ + records/).")
    ap.add_argument("--repo", default=None, help="HF dataset repo id.")
    ap.add_argument("--panel-version", default=None, help="Panel version (latest if omitted).")
    args = ap.parse_args(argv)
    bundle, scores = load_bundle(
        local_dir=args.local, repo=args.repo, panel_version=args.panel_version
    )
    build_demo(bundle, scores).launch()


if __name__ == "__main__":
    main()
