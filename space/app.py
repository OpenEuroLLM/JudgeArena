"""Gradio leaderboard Space (thin renderer over a published bundle)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# When deployed, app.py and render.py sit side-by-side at the Space root
# (`from render import ...`); in this repo they live under `space/` as a
# namespace package (`from space.render import ...`). gradio is imported lazily
# (inside build_demo) so importing this module for tests does not require it.
try:
    from render import (
        available_languages,
        distribution_fig,
        language_bar,
        language_table,
        overview_table,
    )
except ModuleNotFoundError:
    from space.render import (
        available_languages,
        distribution_fig,
        language_bar,
        language_table,
        overview_table,
    )

_DATASET_REPO = "OpenEuroLLM/judgearena-leaderboard"  # default published dataset


def load_bundle(local_dir: str | None = None, repo: str | None = None) -> tuple[dict, pd.DataFrame]:
    """Load (bundle, scores) from a local dir or a HF dataset repo."""
    if local_dir:
        base = Path(local_dir)
        bundle = json.loads((base / "leaderboard.json").read_text())
        scores = pd.read_parquet(base / "scores.parquet")
        return bundle, scores

    from huggingface_hub import hf_hub_download

    repo = repo or _DATASET_REPO
    lb = hf_hub_download(repo_id=repo, repo_type="dataset", filename="leaderboard.json")
    sc = hf_hub_download(repo_id=repo, repo_type="dataset", filename="scores.parquet")
    return json.loads(Path(lb).read_text()), pd.read_parquet(sc)


def build_demo(bundle: dict, scores: pd.DataFrame):  # -> gr.Blocks
    import gradio as gr  # lazy: only needed when actually launching the Space

    panel = bundle.get("panel", {})
    langs = available_languages(bundle)
    models = sorted(scores["model"].unique().tolist()) if len(scores) else []

    header = (
        f"### JudgeArena Leaderboard\n"
        f"Judge: `{panel.get('judge_model', '?')}` · panel `{panel.get('panel_version', '?')}` · "
        f"MAE vs Human-ELO: {panel.get('mae_vs_human', float('nan')):.1f}"
    )

    with gr.Blocks(theme=gr.themes.Soft(), title="JudgeArena Leaderboard") as demo:
        gr.Markdown(header)
        with gr.Tab("Overview"):
            gr.Dataframe(value=overview_table(bundle), interactive=False, wrap=True)
        with gr.Tab("By language"):
            if langs:
                lang_dd = gr.Dropdown(langs, value=langs[0], label="Language")
                lang_tbl = gr.Dataframe(value=language_table(bundle, langs[0]), interactive=False)
                lang_plot = gr.Plot(value=language_bar(bundle, langs[0]))

                def _on_lang(lang):
                    return language_table(bundle, lang), language_bar(bundle, lang)

                lang_dd.change(_on_lang, lang_dd, [lang_tbl, lang_plot])
            else:
                gr.Markdown("_No per-language data in this panel._")
        with gr.Tab("Distributions"):
            if models:
                model_sel = gr.Dropdown(models, value=models[: min(3, len(models))],
                                        multiselect=True, label="Models")
                dist_plot = gr.Plot(value=distribution_fig(scores, models[: min(3, len(models))]))
                model_sel.change(lambda ms: distribution_fig(scores, ms), model_sel, dist_plot)
            else:
                gr.Markdown("_No per-battle scores published yet._")
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="judgearena-space")
    ap.add_argument("--local", default=None, help="Local bundle dir (skips HF download).")
    ap.add_argument("--repo", default=None, help="HF dataset repo id.")
    args = ap.parse_args(argv)
    bundle, scores = load_bundle(local_dir=args.local, repo=args.repo)
    build_demo(bundle, scores).launch()


if __name__ == "__main__":
    main()
