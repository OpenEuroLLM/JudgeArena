"""Tests for judgearena.log – structured logging infrastructure."""

from __future__ import annotations

import argparse
import logging

import pytest

from judgearena.log import (
    _ROOT_LOGGER_NAME,
    attach_file_handler,
    configure_logging,
    get_logger,
    make_run_log_path,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Remove handlers added by configure_logging so tests don't leak state."""
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    yield
    root.handlers.clear()
    root.setLevel(logging.WARNING)


def _console_handler_level() -> int:
    """Return the level of the console (non-file) handler."""
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            return h.level
    raise AssertionError("No console handler found")


# ---------- get_logger ----------


@pytest.mark.parametrize(
    "input_name, expected",
    [
        (None, _ROOT_LOGGER_NAME),
        ("judgearena.utils", "judgearena.utils"),
        ("mymodule", "judgearena.mymodule"),  # bare names get prefixed
    ],
)
def test_get_logger_naming(input_name, expected):
    assert get_logger(input_name).name == expected


# ---------- configure_logging ----------


@pytest.mark.parametrize(
    "verbosity, expected_level",
    [(-1, logging.WARNING), (0, logging.INFO), (1, logging.DEBUG), (3, logging.DEBUG)],
)
def test_configure_logging_verbosity(verbosity, expected_level):
    configure_logging(verbosity)
    assert _console_handler_level() == expected_level


def test_configure_logging_no_duplicate_handlers():
    """Calling configure_logging twice must not add a second console handler."""
    configure_logging(0)
    configure_logging(1)
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    console_handlers = [
        h for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(console_handlers) == 1


def test_env_var_overrides_verbosity(monkeypatch):
    """JUDGEARENA_LOG_LEVEL env-var should override the CLI verbosity flag."""
    monkeypatch.setenv("JUDGEARENA_LOG_LEVEL", "warning")
    configure_logging(1)  # would normally be DEBUG
    assert _console_handler_level() == logging.WARNING


# ---------- file handler ----------


def test_file_handler_captures_debug_even_when_console_is_info(tmp_path):
    """File handler should always capture DEBUG, even if console is INFO."""
    log_file = tmp_path / "debug_capture.log"
    configure_logging(0, log_file=log_file)  # console = INFO

    logger = get_logger("judgearena.test_debug")
    logger.debug("only in file")

    for h in logging.getLogger(_ROOT_LOGGER_NAME).handlers:
        h.flush()

    text = log_file.read_text()
    assert "only in file" in text


def test_attach_file_handler_is_idempotent_for_same_path(tmp_path):
    log_file = tmp_path / "run.log"

    first = attach_file_handler(log_file)
    second = attach_file_handler(log_file)

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]

    assert first is second
    assert len(file_handlers) == 1


def test_attach_file_handler_creates_parent_directory(tmp_path):
    log_file = tmp_path / "nested" / "logs" / "run.log"

    attach_file_handler(log_file)

    assert log_file.parent.exists()


# ---------- resolve_verbosity ----------


def test_resolve_verbosity_quiet_overrides_verbose():
    """When both -q and -v are set, quiet wins (returns -1)."""
    from judgearena.cli_common import resolve_verbosity

    ns = argparse.Namespace(verbose=2, quiet=True)
    assert resolve_verbosity(ns) == -1


# ---------- make_run_log_path ----------


def test_make_run_log_path_format(tmp_path):
    path = make_run_log_path(tmp_path)
    assert path.parent == tmp_path
    assert path.name.startswith("run-")
    assert path.suffix == ".log"
    # Timestamp portion: YYYYMMDD_HHMMSS (15 chars)
    assert len(path.stem) == len("run-") + 15
