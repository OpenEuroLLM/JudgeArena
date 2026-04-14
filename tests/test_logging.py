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


# ---------- get_logger ----------


def test_get_logger_none_returns_root():
    logger = get_logger(None)
    assert logger.name == _ROOT_LOGGER_NAME


def test_get_logger_with_module_name():
    logger = get_logger("judgearena.utils")
    assert logger.name == "judgearena.utils"


def test_get_logger_auto_prefixes_bare_name():
    logger = get_logger("mymodule")
    assert logger.name == "judgearena.mymodule"


def test_get_logger_no_double_prefix():
    """Passing a name that already starts with 'judgearena' is not prefixed again."""
    logger = get_logger("judgearena.evaluate")
    assert logger.name == "judgearena.evaluate"


# ---------- configure_logging ----------


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


def test_configure_logging_default_is_info():
    configure_logging(0)
    assert _console_handler_level() == logging.INFO


def test_configure_logging_quiet():
    configure_logging(-1)
    assert _console_handler_level() == logging.WARNING


def test_configure_logging_verbose():
    configure_logging(1)
    assert _console_handler_level() == logging.DEBUG


def test_configure_logging_extra_verbose():
    """Values > 1 still map to DEBUG."""
    configure_logging(3)
    assert _console_handler_level() == logging.DEBUG


def test_configure_logging_adds_stderr_handler():
    configure_logging()
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    console_handlers = [
        h
        for h in root.handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(console_handlers) == 1


def test_configure_logging_no_duplicate_handlers():
    """Calling configure_logging twice does not add a second handler."""
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


def test_child_logger_inherits_level():
    configure_logging(1)
    child = get_logger("judgearena.sub")
    assert child.getEffectiveLevel() == logging.DEBUG


# ---------- env-var override ----------


def test_env_var_overrides_verbosity(monkeypatch):
    monkeypatch.setenv("JUDGEARENA_LOG_LEVEL", "DEBUG")
    configure_logging(0)  # would normally be INFO
    assert _console_handler_level() == logging.DEBUG


def test_env_var_case_insensitive(monkeypatch):
    monkeypatch.setenv("JUDGEARENA_LOG_LEVEL", "warning")
    configure_logging(1)  # would normally be DEBUG
    assert _console_handler_level() == logging.WARNING


# ---------- attach_file_handler ----------


def test_attach_file_handler(tmp_path):
    configure_logging(0)
    log_file = tmp_path / "test.log"
    fh = attach_file_handler(log_file)
    assert isinstance(fh, logging.FileHandler)
    assert fh.level == logging.DEBUG

    logger = get_logger("judgearena.test_fh")
    logger.info("hello file")
    fh.flush()

    text = log_file.read_text()
    assert "hello file" in text

    # cleanup
    logging.getLogger(_ROOT_LOGGER_NAME).removeHandler(fh)
    fh.close()


def test_configure_logging_with_log_file(tmp_path):
    log_file = tmp_path / "via_configure.log"
    configure_logging(0, log_file=log_file)

    logger = get_logger("judgearena.test_cfg_fh")
    logger.info("configured file handler")

    # flush all handlers
    for h in logging.getLogger(_ROOT_LOGGER_NAME).handlers:
        h.flush()

    text = log_file.read_text()
    assert "configured file handler" in text


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


# ---------- resolve_verbosity ----------


def test_resolve_verbosity_default():
    from judgearena.cli_common import resolve_verbosity

    ns = argparse.Namespace(verbose=0, quiet=False)
    assert resolve_verbosity(ns) == 0


def test_resolve_verbosity_quiet():
    from judgearena.cli_common import resolve_verbosity

    ns = argparse.Namespace(verbose=0, quiet=True)
    assert resolve_verbosity(ns) == -1


def test_resolve_verbosity_verbose():
    from judgearena.cli_common import resolve_verbosity

    ns = argparse.Namespace(verbose=2, quiet=False)
    assert resolve_verbosity(ns) == 2


def test_resolve_verbosity_quiet_overrides_verbose():
    """When both -q and -v are set, quiet wins."""
    from judgearena.cli_common import resolve_verbosity

    ns = argparse.Namespace(verbose=1, quiet=True)
    assert resolve_verbosity(ns) == -1


# ---------- make_run_log_path ----------


def test_make_run_log_path_format(tmp_path):
    path = make_run_log_path(tmp_path)
    assert path.parent == tmp_path
    assert path.name.startswith("run-")
    assert path.suffix == ".log"
    # Timestamp portion is 15 chars: YYYYMMDD_HHMMSS
    stem = path.stem  # e.g. "run-20260414_123456"
    assert len(stem) == len("run-") + 15


def test_make_run_log_path_no_collision(tmp_path):
    """Two calls in the same second produce the same path (by design);
    calls a second apart produce different paths."""
    p1 = make_run_log_path(tmp_path)
    p2 = make_run_log_path(tmp_path)
    # Same second → same timestamp is fine (runs don't start simultaneously)
    assert p1.parent == p2.parent
    assert p1.name.startswith("run-")
