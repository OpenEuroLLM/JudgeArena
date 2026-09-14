"""Tests for judgearena.log – structured logging infrastructure."""

from __future__ import annotations

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


def test_get_logger_naming():
    assert get_logger("mymodule").name == "judgearena.mymodule"


def test_configure_logging_no_duplicate_handlers():
    """Calling configure_logging twice must not add a second console handler."""
    configure_logging(0)
    configure_logging(1)
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    console_handlers = [
        h
        for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(console_handlers) == 1
    assert console_handlers[0].level == logging.DEBUG


def test_env_var_overrides_verbosity(monkeypatch):
    """JUDGEARENA_LOG_LEVEL env-var should override the CLI verbosity flag."""
    monkeypatch.setenv("JUDGEARENA_LOG_LEVEL", "warning")
    configure_logging(1)  # would normally be DEBUG
    assert _console_handler_level() == logging.WARNING


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
    log_file = tmp_path / "nested" / "run.log"

    first = attach_file_handler(log_file)
    second = attach_file_handler(log_file)

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]

    assert log_file.exists()
    assert first is second
    assert len(file_handlers) == 1


def test_quiet_overrides_verbose():
    """When both -q and -v are set, quiet wins (verbosity -1)."""
    from judgearena.config import build_run_config

    cfg = build_run_config(
        [
            "-q",
            "-v",
            "--task",
            "alpaca-eval",
            "--model.name",
            "Dummy/a",
            "--model.baseline",
            "Dummy/b",
            "--judge.model",
            "Dummy/j",
        ]
    )
    assert cfg.run.verbosity == -1


def test_make_run_log_path_format(tmp_path):
    path = make_run_log_path(tmp_path)
    assert path.parent == tmp_path
    assert path.name.startswith("run-")
    assert path.suffix == ".log"
    # Timestamp portion: YYYYMMDD_HHMMSS (15 chars)
    assert len(path.stem) == len("run-") + 15
