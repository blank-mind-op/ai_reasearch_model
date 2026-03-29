# src/utils/logging.py

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

def setup_logging(
        log_file : Path | None  = None,
        level : str = "INFO"
) -> None:
    """
    Call this ONCE at the very start of train.py — before anything else.

    After this call, any file in the project can do:
        from src.utils.logging import get_logger
        log = get_logger(__name__)
        log.info("something.happened", value=42, epoch=3)

    In your terminal (TTY): coloured, human-readable output.
    In CI or a log file:    JSON — one line per event, fully parseable.

    The same code produces both. You never change log calls based on
    where the code runs.
    """
    # These processors run on every log event, in order
    shared_processors = [
        # Merges any context variables you've bound to the logger
        # (useful for adding epoch/run_id to every log line automatically)
        structlog.contextvars.merge_contextvars,

        # Adds the log level to every event: "info", "warning", "error"
        structlog.processors.add_log_level,

        # Adds a timestamp — HH:MM:SS format, readable in terminal
        structlog.processors.TimeStamper(fmt="%H:%M:%S"), # trailing comma is good to add/remove items
    ]

    # Choose renderer based on whether we're in a terminal
    # isatty() returns True in your terminal, False in CI / log files
    if sys.stdout.isatty():
        # Human-readable coloured output for your terminal
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Machine-readable JSON for CI, log files, cloud logging systems
        renderer = structlog.processors.JSONRenderer()
    
    structlog.configure(
        processors=shared_processors + [renderer],
        # make_filtering_bound_logger respects the level you set:
        # INFO = show info/warning/error, hide debug
        # DEBUG = show everything

        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),

        logger_factory=structlog.PrintLoggerFactory(),
        # Caches the logger after first use — small performance win

        cache_logger_on_first_use=True
    )

    # If a log file path is given, also write logs there

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Standard library handler writes to the file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger  = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(getattr(logging, level.upper()))

def get_logger(name : str) -> structlog.BoundLogger:
    """
    Get a logger for a specific module.

    Convention: always call this at module level with __name__:
        log = get_logger(__name__)

    __name__ is automatically the module's full path, e.g.
    "src.training.trainer" — so you always know which file
    a log line came from.
    """

    return structlog.get_logger(name)