"""
Logging utilities for the pokeai package.

Provides:
- setup_logging(): configure a nice console logger (using rich if available)
- get_logger(): get a module-specific logger
- Timer: small context manager to measure durations
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator, Optional

try:
    # Optional: pretty logging if rich is installed
    from rich.logging import RichHandler  # type: ignore
    _HAS_RICH = True
except Exception:  # pragma: no cover
    RichHandler = None
    _HAS_RICH = False


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging for the whole project.
    Call this once (e.g. in CLI or main scripts).
    """
    if logging.getLogger().handlers:
        # Already configured, don't add handlers twice
        return

    if _HAS_RICH and RichHandler is not None:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger for a given module or component.
    If name is None, returns the root logger.
    """
    return logging.getLogger(name)


@contextmanager
def timer(label: str) -> Iterator[None]:
    """
    Context manager to measure elapsed time.

    Example:
        with timer("inference"):
            run_model()
    """
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        logging.getLogger("pokeai.timer").info("%s took %.3f s", label, dt)

