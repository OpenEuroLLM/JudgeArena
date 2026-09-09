"""Inspect content-addressed JudgeArena cache cells."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

import pandas as pd

from judgearena.arenas_utils import extract_turn_text
from judgearena.cache_sqlite import (
    COMPLETION_DB_NAME,
    JUDGEMENT_DB_NAME,
    CompletionCache,
    JudgementCache,
    read_descriptor,
)
from judgearena.datasets import load_battles, load_instructions
from judgearena.evaluate import PairScore
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import EloProtocol, MetaEvalProtocol

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, ScrollableContainer, Vertical
    from textual.widgets import (
        Button,
        Checkbox,
        Footer,
        Header,
        Label,
        Markdown,
        Select,
    )
except ImportError:
    App = None

CacheRole = Literal["completions", "judgements"]


@dataclass(frozen=True)
class CacheCell:
    """One role/task/provider/model/descriptor cache partition."""

    role: CacheRole
    task: str
    provider: str
    model: str
    descriptor_hash: str
    folder: Path
    descriptor: dict

    @property
    def model_spec(self) -> str:
        return f"{self.provider}/{self.model}"

    @property
    def db_path(self) -> Path:
        name = COMPLETION_DB_NAME if self.role == "completions" else JUDGEMENT_DB_NAME
        return self.folder / name

    @property
    def key(self) -> str:
        return f"{self.role}/{self.task}/{self.model_spec}/{self.descriptor_hash}"


def iter_cache_cells(
    store_root: Path | str,
    *,
    role: CacheRole | None = None,
) -> list[CacheCell]:
    """Return valid cache cells in deterministic path order."""
    root = Path(store_root).expanduser()
    roles = (role,) if role is not None else ("completions", "judgements")
    cells = []
    for current_role in roles:
        role_root = root / current_role
        if not role_root.exists():
            continue
        for metadata_path in sorted(role_root.glob("*/*/*/*/metadata.json")):
            folder = metadata_path.parent
            relative = folder.relative_to(role_root)
            task, provider, model, descriptor_hash = relative.parts
            cell = CacheCell(
                role=current_role,
                task=unquote(task),
                provider=unquote(provider),
                model=unquote(model),
                descriptor_hash=descriptor_hash,
                folder=folder,
                descriptor=read_descriptor(folder),
            )
            if cell.db_path.exists():
                cells.append(cell)
    return cells


def load_cache_cell(
    cell: CacheCell,
    *,
    instruction_id: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    """Load rows from one cell using its typed filters."""
    store_type = CompletionCache if cell.role == "completions" else JudgementCache
    with store_type(cell.db_path) as store:
        return store.query(instruction_id=instruction_id, model=model)


def cell_row_count(cell: CacheCell) -> int:
    table = "completions" if cell.role == "completions" else "judgements"
    with sqlite3.connect(cell.db_path) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def list_tasks(store_root: Path | str) -> list[str]:
    return sorted({cell.task for cell in iter_cache_cells(store_root)})


def list_models(store_root: Path | str, task: str | None = None) -> list[str]:
    return sorted(
        {
            cell.model_spec
            for cell in iter_cache_cells(store_root, role="completions")
            if task is None or cell.task == task
        }
    )


def list_judges(store_root: Path | str, task: str | None = None) -> list[str]:
    return sorted(
        {
            cell.model_spec
            for cell in iter_cache_cells(store_root, role="judgements")
            if task is None or cell.task == task
        }
    )


@cache
def load_context(task: str) -> pd.DataFrame:
    """Load instruction and language columns indexed by cache instruction ID."""
    resolved = get_packaged_task(task)
    if resolved is None:
        return pd.DataFrame(columns=["instruction", "language"])
    protocol = resolved.spec.protocol
    if isinstance(protocol, (EloProtocol, MetaEvalProtocol)):
        source = load_battles(resolved)
        index = protocol.arena + ":" + source["question_id"].astype(str)
        instruction = source["conversation_a"].map(
            lambda conversation: extract_turn_text(conversation[0])
        )
    else:
        source = load_instructions(task)
        index = source.index.astype(str)
        column = "instruction" if "instruction" in source else "turn_1"
        instruction = source[column]
    language_column = next(
        (column for column in ("lang", "language") if column in source),
        None,
    )
    language = (
        source[language_column]
        if language_column is not None
        else pd.Series(None, index=source.index)
    )
    return pd.DataFrame(
        {
            "instruction": instruction.astype(str).to_numpy(),
            "language": language.to_numpy(),
        },
        index=index,
    )


def list_languages(store_root: Path | str, task: str) -> list[str]:
    del store_root
    context = load_context(task)
    return sorted(str(value) for value in context["language"].dropna().unique())


def _completion_rows(
    store_root: Path | str,
    *,
    task: str,
) -> pd.DataFrame:
    frames = []
    for cell in iter_cache_cells(store_root, role="completions"):
        if cell.task != task:
            continue
        frame = load_cache_cell(cell)
        frame["descriptor_hash"] = cell.descriptor_hash
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("pushed_at", kind="stable")
        .drop_duplicates(["instruction_id", "model"], keep="last")
    )


def load_subset(
    *,
    store_root: Path | str,
    task: str,
    model: str,
    judge: str | None = None,
    languages: list[str] | None = None,
) -> pd.DataFrame:
    """Join cached judge rows with contexts and the latest model completions."""
    judgement_frames = []
    for cell in iter_cache_cells(store_root, role="judgements"):
        if cell.task != task or (judge is not None and cell.model_spec != judge):
            continue
        frame = load_cache_cell(cell, model=model)
        frame["judge_descriptor_hash"] = cell.descriptor_hash
        judgement_frames.append(frame)
    if not judgement_frames:
        return pd.DataFrame()

    rows = pd.concat(judgement_frames, ignore_index=True)
    completions = _completion_rows(store_root, task=task)
    completion_map = (
        completions.set_index(["instruction_id", "model"])["completion"].to_dict()
        if not completions.empty
        else {}
    )
    rows["completion_a"] = [
        completion_map.get((str(row.instruction_id), row.model_a))
        for row in rows.itertuples()
    ]
    rows["completion_b"] = [
        completion_map.get((str(row.instruction_id), row.model_b))
        for row in rows.itertuples()
    ]
    context = load_context(task)
    rows["instruction"] = rows["instruction_id"].astype(str).map(context["instruction"])
    rows["language"] = rows["instruction_id"].astype(str).map(context["language"])
    if languages:
        rows = rows.loc[rows["language"].isin(languages)]

    parser = PairScore()
    rows["preference"] = [
        (
            parsed.preference
            if (parsed := parser.parse_result(completion)) is not None
            else None
        )
        for completion in rows["judge_completion"]
    ]
    return rows.reset_index(drop=True)


def render_row(row: pd.Series, *, show_judge_input: bool = False) -> str:
    """Render one joined cache row for terminal and notebook browsers."""

    def display_value(value: object) -> str:
        return "*(not found)*" if value is None or pd.isna(value) else str(value)

    preference = row.get("preference")
    verdict = (
        "Could not parse"
        if pd.isna(preference)
        else f"Preference for B: {float(preference):.2f}"
    )
    sections = [
        f"**Model A:** `{row['model_a']}`  \n**Model B:** `{row['model_b']}`",
        f"### Instruction\n{display_value(row.get('instruction'))}",
        f"### Completion A\n{display_value(row.get('completion_a'))}",
        f"### Completion B\n{display_value(row.get('completion_b'))}",
        f"### Judge output\n**{verdict}**\n\n{row['judge_completion']}",
    ]
    if show_judge_input:
        sections.append(f"### Judge input\n{row['judge_input']}")
    return "\n\n---\n\n".join(sections)


if App is not None:

    class BrowserApp(App):
        """Interactive terminal browser for cached judge rows."""

        TITLE = "JudgeArena Cache Browser"
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("left", "previous", "Previous", show=False),
            Binding("right", "next", "Next", show=False),
        ]

        def __init__(self, store_root: Path | str):
            super().__init__()
            self.store_root = Path(store_root).expanduser()
            self.rows = pd.DataFrame()
            self.row_index = 0

        def compose(self) -> ComposeResult:
            tasks = list_tasks(self.store_root)
            yield Header()
            with Horizontal():
                with Vertical(id="sidebar"):
                    yield Select([(task, task) for task in tasks], id="task")
                    yield Select([], prompt="Model", id="model")
                    yield Select([], prompt="Judge", id="judge", allow_blank=True)
                    yield Select([], prompt="Language", id="language", allow_blank=True)
                    yield Button("Load", id="load")
                    with Horizontal():
                        yield Button("◀", id="previous")
                        yield Label("—", id="position")
                        yield Button("▶", id="next")
                    yield Checkbox("Show judge input", id="show-input")
                with ScrollableContainer():
                    yield Markdown("*Select a task and model.*", id="content")
            yield Footer()

        def on_select_changed(self, event: Select.Changed) -> None:
            if event.select.id != "task" or event.value is Select.BLANK:
                return
            task = str(event.value)
            self.query_one("#model", Select).set_options(
                [(value, value) for value in list_models(self.store_root, task)]
            )
            self.query_one("#judge", Select).set_options(
                [(value, value) for value in list_judges(self.store_root, task)]
            )
            self.query_one("#language", Select).set_options(
                [(value, value) for value in list_languages(self.store_root, task)]
            )

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "load":
                self._load()
            elif event.button.id == "previous":
                self.action_previous()
            elif event.button.id == "next":
                self.action_next()

        def on_checkbox_changed(self, _event: Checkbox.Changed) -> None:
            self._show()

        def action_previous(self) -> None:
            self.row_index = max(0, self.row_index - 1)
            self._show()

        def action_next(self) -> None:
            self.row_index = min(max(0, len(self.rows) - 1), self.row_index + 1)
            self._show()

        def _select_value(self, selector: str) -> str | None:
            value = self.query_one(selector, Select).value
            return None if value is Select.BLANK else str(value)

        def _load(self) -> None:
            task = self._select_value("#task")
            model = self._select_value("#model")
            if task is None or model is None:
                return
            language = self._select_value("#language")
            self.rows = load_subset(
                store_root=self.store_root,
                task=task,
                model=model,
                judge=self._select_value("#judge"),
                languages=[language] if language is not None else None,
            )
            self.row_index = 0
            self._show()

        def _show(self) -> None:
            content = self.query_one("#content", Markdown)
            position = self.query_one("#position", Label)
            if self.rows.empty:
                position.update("—")
                content.update("*No matching rows.*")
                return
            position.update(f"{self.row_index + 1}/{len(self.rows)}")
            content.update(
                render_row(
                    self.rows.iloc[self.row_index],
                    show_judge_input=self.query_one("#show-input", Checkbox).value,
                )
            )


def launch_browser(store_root: Path | str) -> None:
    if App is None:
        raise ImportError("Install JudgeArena's browser extra to use the terminal UI.")
    BrowserApp(store_root).run()


def _select_cells(cells: list[CacheCell], args: argparse.Namespace) -> list[CacheCell]:
    return [
        cell
        for cell in cells
        if (args.task is None or cell.task == args.task)
        and (args.provider is None or cell.provider == args.provider)
        and (args.model is None or cell.model == args.model)
        and (
            args.descriptor is None or cell.descriptor_hash.startswith(args.descriptor)
        )
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--role", choices=("completions", "judgements"))
    parser.add_argument("--task")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--descriptor")
    parser.add_argument("--instruction-id")
    parser.add_argument("--candidate-model")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--list", action="store_true", dest="list_cells")
    parser.add_argument("--interactive", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.interactive or not any(
        (
            args.role,
            args.task,
            args.provider,
            args.model,
            args.descriptor,
            args.instruction_id,
            args.candidate_model,
            args.list_cells,
        )
    ):
        launch_browser(args.store_root)
        return
    cells = _select_cells(
        iter_cache_cells(args.store_root, role=args.role),
        args,
    )
    if args.list_cells or len(cells) != 1:
        for cell in cells:
            print(
                json.dumps(
                    {
                        "cell": cell.key,
                        "rows": cell_row_count(cell),
                        "descriptor": cell.descriptor,
                    },
                    sort_keys=True,
                )
            )
        return

    rows = load_cache_cell(
        cells[0],
        instruction_id=args.instruction_id,
        model=args.candidate_model,
    )
    print(rows.head(args.limit).to_json(orient="records", force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
