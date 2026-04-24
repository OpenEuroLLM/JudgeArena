"""Structured logging for judgearena.

Provides a thin wrapper around Python's ``logging`` module so that every
module in the package can do::

    from judgearena.log import get_logger
    logger = get_logger(__name__)
    logger.info("Loaded %d instructions", n)

Call :func:`configure_logging` once from the CLI entrypoint to set the
verbosity level based on ``-v`` / ``-q`` flags.  The default level is
``INFO``, which mirrors the current ``print()`` behaviour.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_ROOT_LOGGER_NAME = "judgearena"

_CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _resolve_log_path(log_file: str | Path) -> Path:
    return Path(log_file).expanduser().resolve()


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the ``judgearena`` namespace.

    Typical usage at the top of a module::

        logger = get_logger(__name__)

    The returned logger inherits its level and handlers from the root
    ``judgearena`` logger configured by :func:`configure_logging`.
    """
    if name is None:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    # Strip the common prefix so child loggers are short
    # e.g. "judgearena.utils" → child of "judgearena"
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def configure_logging(
    verbosity: int = 0,
    log_file: str | Path | None = None,
) -> None:
    """Set up the root ``judgearena`` logger.

    Args:
        verbosity: Controls the console log level.

            * ``-1``  (``-q``)  → ``WARNING`` – only warnings and errors.
            * ``0``   (default) → ``INFO``    – progress messages (mirrors
              the old ``print()`` behaviour).
            * ``1+`` (``-v``)   → ``DEBUG``   – verbose: data previews,
              cache hits, first-completion dumps, etc.

            The environment variable ``JUDGEARENA_LOG_LEVEL`` overrides the
            CLI flags when set (e.g. ``JUDGEARENA_LOG_LEVEL=DEBUG``).

        log_file: Optional path to a log file.  When provided, a
            ``FileHandler`` at ``DEBUG`` level is attached so that the full
            trace is always available on disk regardless of the console
            verbosity.
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)

    # Resolve level: env-var beats CLI flags.
    if verbosity < 0:
        level = logging.WARNING
    elif verbosity == 0:
        level = logging.INFO
    else:
        level = logging.DEBUG

    env_level = os.environ.get("JUDGEARENA_LOG_LEVEL")
    if env_level:
        level = getattr(logging, env_level.upper(), level)

    root.setLevel(min(level, logging.DEBUG))  # root allows everything; handlers filter

    # --- console handler ---
    # Avoid duplicate handlers when configure_logging is called more than once
    # (e.g. in tests).
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(_CONSOLE_FORMAT, datefmt=_CONSOLE_DATEFMT)
        )
        root.addHandler(handler)
    else:
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                h.setLevel(level)

    # --- file handler (explicit --log-file) ---
    if log_file is not None:
        attach_file_handler(log_file)


def attach_file_handler(
    log_file: str | Path,
    level: int = logging.DEBUG,
) -> logging.FileHandler:
    """Attach a file handler to the root ``judgearena`` logger.

    This is called automatically by :func:`configure_logging` when
    ``log_file`` is provided, and can also be called later — for example
    once the result folder is known — to save a ``run.log`` artifact
    alongside evaluation results.

    Returns the handler so the caller can remove it if needed.
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    resolved_log_file = _resolve_log_path(log_file)

    for handler in root.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        handler_path = getattr(handler, "baseFilename", None)
        if handler_path is None:
            continue
        if Path(handler_path).resolve() == resolved_log_file:
            handler.setLevel(level)
            handler.setFormatter(
                logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATEFMT)
            )
            return handler

    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(resolved_log_file)
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATEFMT)
    )
    root.addHandler(fh)
    return fh


def make_run_log_path(folder: str | Path) -> Path:
    """Return a timestamped log path like ``folder/run-20260414_123456.log``.

    The timestamp ensures multiple runs never overwrite each other.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(folder) / f"run-{ts}.log"
