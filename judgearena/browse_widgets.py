"""Notebook widgets for browsing content-addressed cache rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import ipywidgets as widgets
    from IPython.display import Markdown, clear_output, display
except ImportError:
    widgets = None
    Markdown = None
    clear_output = None
    display = None

from judgearena.browse_cache import (
    list_judges,
    list_languages,
    list_models,
    list_tasks,
    load_subset,
    render_row,
)


def make_ui(store_root: Path | str) -> Any:
    """Display selectors and rows for the cache cells under ``store_root``."""
    if widgets is None or Markdown is None or clear_output is None or display is None:
        raise ImportError("Install ipywidgets and IPython to use the cache browser.")

    root = Path(store_root).expanduser()
    task = widgets.Dropdown(options=list_tasks(root), description="Task:")
    model = widgets.Dropdown(description="Model:")
    judge = widgets.Dropdown(description="Judge:")
    language = widgets.Dropdown(description="Language:")
    show_input = widgets.Checkbox(description="Show judge input")
    previous = widgets.Button(description="◀")
    next_button = widgets.Button(description="▶")
    position = widgets.Label("—")
    output = widgets.Output()
    state = {"rows": None, "index": 0}

    def render() -> None:
        with output:
            clear_output()
            rows = state["rows"]
            if rows is None or rows.empty:
                position.value = "—"
                display(Markdown("*No matching rows.*"))
                return
            position.value = f"{state['index'] + 1}/{len(rows)}"
            display(
                Markdown(
                    render_row(
                        rows.iloc[state["index"]],
                        show_judge_input=show_input.value,
                    )
                )
            )

    def refresh_options(_event=None) -> None:
        if task.value is None:
            return
        model.options = list_models(root, task.value)
        judge.options = [None, *list_judges(root, task.value)]
        language.options = [None, *list_languages(root, task.value)]

    def load(_event=None) -> None:
        if task.value is None or model.value is None:
            return
        state["rows"] = load_subset(
            store_root=root,
            task=task.value,
            model=model.value,
            judge=judge.value,
            languages=[language.value] if language.value is not None else None,
        )
        state["index"] = 0
        render()

    def move(delta: int) -> None:
        rows = state["rows"]
        if rows is None or rows.empty:
            return
        state["index"] = max(0, min(len(rows) - 1, state["index"] + delta))
        render()

    task.observe(refresh_options, names="value")
    for selector in (model, judge, language):
        selector.observe(load, names="value")
    show_input.observe(lambda _event: render(), names="value")
    previous.on_click(lambda _event: move(-1))
    next_button.on_click(lambda _event: move(1))

    controls = widgets.VBox(
        [
            widgets.HBox([task, model]),
            widgets.HBox([judge, language, show_input]),
            widgets.HBox([previous, position, next_button]),
        ]
    )
    ui = widgets.VBox([controls, output])
    refresh_options()
    load()
    display(ui)
    return ui
