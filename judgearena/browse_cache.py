"""Functions and TUI for browsing the JudgeArena annotation cache.

CLI usage:
    uv run python judgearena/browse_cache.py --store-root ~/judgearena-data/db

The annotation cache lives under a store root with this layout:
    completions/{arena}/{model_name}/{provider}/
        completions.db   — instruction_index → completion
        metadata.json    — arena, model, languages, n_per_lang, ...
    judgements/{arena}/{judge_name}/{provider}/
        judgements.db    — instruction_index, model_A, model_B, judge_output
        metadata.json    — judge model
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from judgearena.arenas_utils import _extract_instruction_text, load_arena_dataframe
from judgearena.store_sqlite import SQLiteCompletionStore, SQLiteJudgementStore

STORE_ROOT = Path.home() / "judgearena-data" / "db"
_DISK_CACHE_DIR = STORE_ROOT / ".cache"

ALL_LANGUAGES = sorted(
    [
        "ar",
        "bn",
        "ca",
        "cs",
        "de",
        "el",
        "en",
        "eo",
        "es",
        "fa",
        "fi",
        "fr",
        "he",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "ko",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sl",
        "sr",
        "sv",
        "th",
        "tr",
        "uk",
        "vi",
        "zh",
    ]
)

_ARENA_CACHE: dict[str, pd.DataFrame] = {}


def _extract_response(conv, turn: int) -> str:
    """Extract plain text from a conversation turn (handles str and block-array content)."""
    try:
        content = conv[turn]["content"]
        if isinstance(content, str):
            return content
        return " ".join(b["text"] for b in content if b.get("type") == "text")
    except Exception:
        return ""


def load_context(
    arena: str = "LMArena+ComparIA",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Return a slim arena DataFrame (instruction, response_a/b, lang, model_a/b).

    Loading order:
      1. In-process memory cache (instant).
      2. Disk parquet cache under ``cache_dir`` (~3 s, written on first use).
      3. HuggingFace download + extraction (~20 s, only on very first run).
    """
    if arena in _ARENA_CACHE:
        return _ARENA_CACHE[arena]

    _cache_dir = (cache_dir or _DISK_CACHE_DIR).expanduser()

    cache_file = _cache_dir / f"{arena}.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        df = load_arena_dataframe(arena)
        # extract instruction and keep only some columns
        df = pd.DataFrame(
            {
                "instruction": df["conversation_a"].apply(
                    lambda c: _extract_instruction_text(c[0])
                ),
                "response_a": df["conversation_a"].apply(
                    lambda c: _extract_response(c, 1)
                ),
                "response_b": df["conversation_b"].apply(
                    lambda c: _extract_response(c, 1)
                ),
                "lang": df["lang"],
                "model_a": df["model_a"],
                "model_b": df["model_b"],
            }
        )
        _cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
    _ARENA_CACHE[arena] = df
    return df


def list_models(store_root: Path = STORE_ROOT) -> list[str]:
    """Return sorted model specs available in the completions store."""
    models: set[str] = set()
    comp_dir = store_root / "completions"
    if not comp_dir.exists():
        return []
    for meta_path in comp_dir.rglob("metadata.json"):
        try:
            m = json.loads(meta_path.read_text()).get("model")
            if m:
                models.add(m)
        except Exception:
            pass
    return sorted(models)


def list_languages() -> list[str]:
    """Return the hardcoded list of supported languages."""
    return list(ALL_LANGUAGES)


def list_judges(store_root: Path = STORE_ROOT) -> list[str]:
    """Return sorted judge model specs available in the judgements store."""
    judges: set[str] = set()
    judge_dir = store_root / "judgements"
    if not judge_dir.exists():
        return []
    for meta_path in judge_dir.rglob("metadata.json"):
        try:
            m = json.loads(meta_path.read_text()).get("model")
            if m:
                judges.add(m)
        except Exception:
            pass
    return sorted(judges)


def _apply_filters(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Apply the arena filter config from a metadata dict to a raw arena DF."""
    langs = meta.get("languages")
    if langs and langs != "all":
        lang_list = langs.split("-") if isinstance(langs, str) else list(langs)
        df = df[df["lang"].isin(lang_list)]
    n_per_lang = meta.get("n_instructions_per_language")
    if n_per_lang is not None:
        df = df.groupby("lang").head(int(n_per_lang))
    n_total = meta.get("n_instructions")
    if n_total is not None:
        df = df.head(int(n_total))
    return df.reset_index(drop=True)


def _load_model_meta(model: str, arena: str, store_root: Path) -> dict | None:
    """Find the completions metadata dict for a specific model+arena combo."""
    comp_dir = store_root / "completions" / arena
    if not comp_dir.exists():
        return None
    for meta_path in comp_dir.rglob("metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("model") == model:
                return meta
        except Exception:
            pass
    return None


def _load_our_completions(model: str, arena: str, store_root: Path) -> dict[int, str]:
    """Return {instruction_index: completion} for our model from the completions DB."""
    comp_dir = store_root / "completions" / arena
    if not comp_dir.exists():
        return {}
    for meta_path in comp_dir.rglob("metadata.json"):
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("model") == model:
                db_path = meta_path.parent / "completions.db"
                if db_path.exists():
                    with SQLiteCompletionStore(db_path) as s:
                        df = s.query()
                    return dict(
                        zip(
                            df["instruction_index"].astype(int),
                            df["completion"],
                            strict=True,
                        )
                    )
        except Exception:
            pass
    return {}


_MULTILINGUAL_LANGS = sorted(
    [
        "ar",
        "bn",
        "ca",
        "cs",
        "de",
        "el",
        "en",
        "eo",
        "es",
        "fa",
        "fi",
        "fr",
        "he",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "ko",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sl",
        "sr",
        "sv",
        "th",
        "tr",
        "uk",
        "vi",
        "zh",
    ]
)

_KNOWN_PROVIDERS = ("VLLM", "OpenRouter", "LlamaCpp")


def _judge_csv_stem(p: Path) -> str:
    name = p.name
    for ext in (".csv.zip", ".csv"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return p.stem


def _model_from_stem(stem: str, known_hashes: dict[str, str]) -> str | None:
    """Return model spec for a judge CSV stem (hash or structured name), or None."""
    bare = stem.removeprefix("judge_")
    if len(bare) == 16 and re.fullmatch(r"[0-9a-f]+", bare):
        return known_hashes.get(bare)
    # Structured: LMArena+ComparIA_VLLM_Qwen_Qwen2.5-7B-Instruct_500_None_en_...
    for provider in _KNOWN_PROVIDERS:
        marker = f"_{provider}_"
        idx = bare.find(marker)
        if idx == -1:
            continue
        rest = bare[idx + len(marker) :]
        parts = rest.split("_")
        if len(parts) < 5:
            continue
        model_slug = "_".join(parts[:-5])
        if model_slug:
            # best-effort: restore slashes — works for most HF model paths
            return f"{provider}/{model_slug.replace('_', '/', 1)}"
    return None


def _build_hash_map(models: list[str] | None = None) -> dict[str, str]:
    """Return {hex16_hash: model_spec} for all known multilingual runs."""
    if models is None:
        models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3.5-9B",
            "allenai/Olmo-3-1025-7B",
            "allenai/Olmo-3-7B-Instruct",
            "allenai/Olmo-3-7B-Think",
            "openeurollm/datamix-9b-Dolci-Translated-A-75EN",
            "swiss-ai/Apertus-8B-Instruct-2509",
            "utter-project/EuroLLM-9B-Instruct-2512",
        ]
    langs_str = "-".join(_MULTILINGUAL_LANGS)
    result = {}
    for model in models:
        slug = f"VLLM_{model.replace('/', '_')}"
        for template in [
            f"LMArena+ComparIA_{slug}_None_100_{langs_str}_8192_32768",
            f"LMArena+ComparIA_{slug}_500_None_{langs_str}_8192_32768",
            f"LMArena+ComparIA_{slug}_None_100_{langs_str}_500_1000_max_model_len=2048",
        ]:
            h = hashlib.sha256(template.encode()).hexdigest()[:16]
            result[h] = f"VLLM/{model}"
    return result


_HASH_MAP = _build_hash_map()


def _find_judge_csvs(model: str, csv_dir: Path) -> list[Path]:
    """Return all judge CSV paths in csv_dir that correspond to model."""
    matches: list[Path] = []
    for p in sorted(csv_dir.glob("judge_*.csv*")):
        stem = _judge_csv_stem(p)
        resolved = _model_from_stem(stem, _HASH_MAP)
        if resolved == model:
            matches.append(p)
    return matches


def _read_judge_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_subset_from_csv(
    model: str,
    csv_dir: Path,
    languages: list[str] | None = None,
) -> pd.DataFrame:
    """Load annotation rows directly from judge CSV files (correct instruction text).

    This bypasses the arena DF lookup, which can be misaligned when the HF
    dataset has grown since the completions/judgements were generated.
    Row position in the judge CSV = instruction_index.
    """
    empty = pd.DataFrame(
        columns=[
            "instruction_index",
            "source",
            "instruction",
            "lang",
            "model_a",
            "model_b",
            "completion_a",
            "completion_b",
            "preference",
            "judgement",
        ]
    )
    csv_paths = _find_judge_csvs(model, csv_dir)
    if not csv_paths:
        return empty

    dfs: list[pd.DataFrame] = []
    for path in csv_paths:
        raw = _read_judge_csv(path)
        rows = []
        for i, row in raw.iterrows():
            is_ours_a = bool(row.get("our_model_is_position_a", True))
            opponent = str(row.get("opponent_model", ""))
            model_a = model if is_ours_a else opponent
            model_b = opponent if is_ours_a else model
            comp_a = str(row.get("completion_A") or "")
            comp_b = str(row.get("completion_B") or "")
            pref_raw = row.get("pref")
            pref = (
                float(pref_raw)
                if pref_raw is not None and str(pref_raw) not in ("", "nan")
                else None
            )
            rows.append(
                {
                    "instruction_index": int(i),
                    "source": path.name,
                    "instruction": str(row.get("instruction") or ""),
                    "lang": "",  # filled below
                    "model_a": model_a,
                    "model_b": model_b,
                    "completion_a": comp_a,
                    "completion_b": comp_b,
                    "preference": pref,
                    "judgement": str(row.get("judge_completion") or ""),
                }
            )
        dfs.append(pd.DataFrame(rows))

    if not dfs:
        return empty

    df = pd.concat(dfs, ignore_index=True)

    if languages is not None:
        arena_df = load_context()
        lang_map: dict[str, str] = dict(
            zip(arena_df["instruction"], arena_df["lang"], strict=True)
        )
        df["lang"] = df["instruction"].map(lang_map).fillna("")
        df = df[df["lang"].isin(languages)].reset_index(drop=True)

    return df


class _PairScore:
    temperature = 0.3

    def parse(self, judge_output: str | None) -> float | None:
        if not judge_output:
            return None
        s = judge_output.lower()
        ma = re.search(r'score.*?a[": *\n]*(-?\d+)', s)
        mb = re.search(r'score.*?b[": *\n]*(-?\d+)', s)
        if ma is None or mb is None:
            return None
        sa, sb = float(ma.group(1)), float(mb.group(1))
        return float(
            1
            - np.exp(self.temperature * sa)
            / np.exp(self.temperature * np.array([sa, sb])).sum()
        )


_score_parser = _PairScore()


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def load_subset(
    model: str,
    languages: list[str] | None = None,
    judge: str | None = None,
    arena: str = "LMArena+ComparIA",
    store_root: Path = STORE_ROOT,
) -> pd.DataFrame:
    """Load annotation rows for a given model.

    Args:
        model: Model spec, e.g. ``"VLLM/Qwen/Qwen3-8B"``.
        languages: Keep only these ISO-639-1 language codes (``None`` = all).
        judge: Keep only annotations from this judge model (``None`` = all).
        arena: Arena name (default ``"LMArena+ComparIA"``).
        store_root: Root of the SQLite store (default ``~/judgearena-data/db``).

    Returns:
        DataFrame with columns:
            instruction, model_a, model_b, completion_a, completion_b,
            preference, judgement
    """
    empty = pd.DataFrame(
        columns=[
            "instruction",
            "model_a",
            "model_b",
            "completion_a",
            "completion_b",
            "preference",
            "judgement",
        ]
    )

    # arena DF (module-level cache, loaded once)
    raw_df = load_context(arena)
    meta = _load_model_meta(model, arena, store_root)
    filtered_df = (
        _apply_filters(raw_df, meta) if meta else raw_df.reset_index(drop=True)
    )

    # our model's completions (single DB read)
    our_completions = _load_our_completions(model, arena, store_root)

    # judgements DB(s)
    judge_dir = store_root / "judgements" / arena
    if not judge_dir.exists():
        return empty

    dfs: list[pd.DataFrame] = []
    for meta_path in judge_dir.rglob("metadata.json"):
        try:
            jmeta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if judge is not None and jmeta.get("model") != judge:
            continue
        db_path = meta_path.parent / "judgements.db"
        if not db_path.exists():
            continue
        with SQLiteJudgementStore(db_path) as s:
            df = s.query(model=model)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return empty

    raw = pd.concat(dfs, ignore_index=True)

    # join arena DF for instruction text and opponent completion
    rows: list[dict] = []
    for _, row in raw.iterrows():
        idx = int(row["instruction_index"])
        if idx < 0 or idx >= len(filtered_df):
            continue

        arena_row = filtered_df.iloc[idx]
        lang = arena_row.get("lang", "")
        if languages is not None and lang not in languages:
            continue

        instruction = arena_row["instruction"]

        model_a = row["model_A"]
        model_b = row["model_B"]
        our_model_is_a = model_a == model
        opponent = model_b if our_model_is_a else model_a

        # opponent completion: response_a if opponent was arena's model_a, else response_b
        opp_completion = (
            arena_row["response_a"]
            if arena_row["model_a"] == opponent
            else arena_row["response_b"]
        )

        our_completion = our_completions.get(idx, "")
        completion_a = our_completion if our_model_is_a else opp_completion
        completion_b = opp_completion if our_model_is_a else our_completion

        rows.append(
            {
                "instruction": instruction,
                "model_a": model_a,
                "model_b": model_b,
                "completion_a": completion_a,
                "completion_b": completion_b,
                "preference": _score_parser.parse(row.get("judge_output")),
                "judgement": row.get("judge_output", ""),
            }
        )

    return pd.DataFrame(rows) if rows else empty


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

_CSS = """
Screen {
    layout: horizontal;
}

#sidebar {
    width: 36;
    height: 100%;
    border-right: solid $primary-darken-2;
    background: $panel;
    padding: 1 1;
}

#sidebar Label {
    margin-top: 1;
    color: $text-muted;
    text-style: bold;
}

#nav-row {
    height: 3;
    align: center middle;
    margin-top: 1;
}

#nav-row Button {
    min-width: 5;
}

#nav-label {
    width: 1fr;
    text-align: center;
    color: $text;
}

#content {
    padding: 1 2;
    height: 100%;
}

#loading-label {
    color: $text-muted;
    text-style: italic;
}
"""


def _fmt_pref(pref) -> str:
    if pref is None or (isinstance(pref, float) and np.isnan(pref)):
        return "*(could not parse)*"
    if pref < 0.45:
        return f"**A wins** (pref = {pref:.2f})"
    if pref > 0.55:
        return f"**B wins** (pref = {pref:.2f})"
    return f"Tie (pref = {pref:.2f})"


def _render_row(row: pd.Series, show_judge_input: bool) -> str:
    model_a = row["model_a"]
    model_b = row["model_b"]
    instruction = (row.get("instruction") or "").strip()
    completion_a = (row.get("completion_a") or "").strip()
    completion_b = (row.get("completion_b") or "").strip()
    judgement = (row.get("judgement") or "").strip()
    pref_str = _fmt_pref(row.get("preference"))

    parts = [
        f"**Model A:** `{model_a}`  \n**Model B:** `{model_b}`",
        "---",
        "### Instruction",
        instruction or "*(empty)*",
        "---",
        f"### Completion A\n*{model_a}*",
        completion_a or "*(not found in completions store)*",
        "---",
        f"### Completion B\n*{model_b}*",
        completion_b or "*(not found in completions store)*",
        "---",
        "### Judge verdict",
        pref_str,
        "",
        judgement,
    ]

    if show_judge_input:
        parts += [
            "---",
            "### Judge input",
            "*(not stored — judge_input was not saved during migration)*",
        ]

    return "\n\n".join(parts)


try:
    from textual import work
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

    class BrowserApp(App):
        CSS = _CSS
        TITLE = "JudgeArena Browser"
        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("left", "prev", "Previous", show=False),
            Binding("right", "next", "Next", show=False),
        ]

        def __init__(
            self,
            models: list[str],
            judges: list[str],
            store_root: Path = STORE_ROOT,
            csv_dir: Path | None = None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self._models = models
            self._judges = judges
            self._store_root = store_root
            self._csv_dir = csv_dir
            self._df: pd.DataFrame = pd.DataFrame()
            self._idx: int = 0

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal():
                with Vertical(id="sidebar"):
                    yield Label("Model")
                    yield Select(
                        [(m, m) for m in self._models],
                        id="model-sel",
                        prompt="Select model…",
                    )
                    yield Label("Judge")
                    yield Select(
                        [(j, j) for j in self._judges],
                        id="judge-sel",
                        prompt="All judges",
                        allow_blank=True,
                    )
                    yield Label("Language")
                    yield Select(
                        [(lg, lg) for lg in list_languages()],
                        id="lang-sel",
                        prompt="All languages",
                        allow_blank=True,
                    )
                    yield Label("Index")
                    with Horizontal(id="nav-row"):
                        yield Button("◀", id="prev-btn", disabled=True)
                        yield Label("—", id="nav-label")
                        yield Button("▶", id="next-btn", disabled=True)
                    yield Checkbox("Show judge input", False, id="show-judge-input")
                with ScrollableContainer(id="content"):
                    yield Markdown("*Select a model to begin.*", id="display")
            yield Footer()

        # ------------------------------------------------------------------
        # Event handlers
        # ------------------------------------------------------------------

        def on_select_changed(self, event: Select.Changed) -> None:
            if event.select.id == "model-sel" and event.value is Select.BLANK:
                return
            # Read all values on the main thread, then hand off to worker
            self._trigger_load()

        def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
            if event.checkbox.id == "show-judge-input":
                self._show()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "prev-btn":
                self._go(-1)
            elif event.button.id == "next-btn":
                self._go(1)

        def action_prev(self) -> None:
            self._go(-1)

        def action_next(self) -> None:
            self._go(1)

        # ------------------------------------------------------------------
        # Internal helpers
        # ------------------------------------------------------------------

        def _read_filters(self) -> tuple[str | None, str | None, list[str] | None]:
            """Return (model, judge, languages) — must be called on the main thread."""

            def _val(widget_id: str) -> str | None:
                v = self.query_one(widget_id, Select).value
                return None if (not v or v is Select.BLANK) else v

            model = _val("#model-sel")
            judge = _val("#judge-sel")
            lang = _val("#lang-sel")
            languages = [lang] if lang else None
            return model, judge, languages

        def _trigger_load(self) -> None:
            model, judge, languages = self._read_filters()
            if model is None:
                return
            self.query_one("#display", Markdown).update("*Loading…*")
            self._load(model, judge, languages)

        @work(thread=True, exclusive=True, exit_on_error=False)
        def _load(
            self, model: str, judge: str | None, languages: list[str] | None
        ) -> None:
            try:
                if self._csv_dir is not None:
                    df = load_subset_from_csv(
                        model=model,
                        csv_dir=self._csv_dir,
                        languages=languages,
                    )
                else:
                    df = load_subset(
                        model=model,
                        judge=judge,
                        languages=languages,
                        store_root=self._store_root,
                    )
                self.call_from_thread(self._on_loaded, df)
            except Exception as exc:
                self.call_from_thread(self._on_error, str(exc))

        def _on_loaded(self, df: pd.DataFrame) -> None:
            self._df = df
            self._idx = 0
            self._refresh_nav()
            self._show()

        def _on_error(self, message: str) -> None:
            md = self.query_one("#display", Markdown)
            md.update(f"*Error loading data:*\n\n```\n{message}\n```")

        def _go(self, delta: int) -> None:
            if self._df.empty:
                return
            self._idx = max(0, min(len(self._df) - 1, self._idx + delta))
            self._refresh_nav()
            self._show()

        def _refresh_nav(self) -> None:
            total = len(self._df)
            label = self.query_one("#nav-label", Label)
            label.update(f"{self._idx + 1}/{total}" if total else "—")
            self.query_one("#prev-btn", Button).disabled = self._idx <= 0 or total == 0
            self.query_one("#next-btn", Button).disabled = self._idx >= total - 1

        def _show(self) -> None:
            md = self.query_one("#display", Markdown)
            if self._df.empty:
                md.update("*No data for the selected filters.*")
                return
            show_ji = self.query_one("#show-judge-input", Checkbox).value
            md.update(_render_row(self._df.iloc[self._idx], show_judge_input=show_ji))

    _TEXTUAL_AVAILABLE = True

except ImportError:
    _TEXTUAL_AVAILABLE = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import os

    _CSV_CANDIDATES = [
        Path(__file__).parent.parent / "slurmpilot_scripts" / "olmo3-evals" / "data",
        Path(__file__).parent.parent / "slurmpilot_scripts" / "elaine_evals",
    ]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store-root",
        type=Path,
        default=STORE_ROOT,
        help="Root directory of the SQLite store (default: ~/judgearena-data/db)",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Directory with judge CSV/CSV.zip files for correct instruction alignment.",
    )
    args = parser.parse_args()
    store_root = args.store_root.expanduser()

    if args.csv_dir is not None:
        csv_dir: Path | None = args.csv_dir.expanduser()
    else:
        csv_dir = next((p for p in _CSV_CANDIDATES if p.is_dir()), None)

    if not store_root.exists():
        parser.error(f"Store root not found: {store_root}")

    if not _TEXTUAL_AVAILABLE:
        print("Textual is not installed. Run: uv add textual")
        return

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    print("Loading arena data (cached after first run)…", flush=True)
    load_context()  # populate module-level cache before starting the TUI

    models = list_models(store_root)
    judges = list_judges(store_root)

    if not models:
        print(f"No completions found under {store_root}")
        return

    if csv_dir is not None:
        print(f"Using judge CSVs from {csv_dir} for correct instruction alignment.")
    print(f"Found {len(models)} model(s), {len(judges)} judge(s). Starting browser…\n")
    BrowserApp(
        models=models, judges=judges, store_root=store_root, csv_dir=csv_dir
    ).run()


if __name__ == "__main__":
    main()
