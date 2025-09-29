"""Configuration helpers for the CAD Quoter application."""
from __future__ import annotations

import logging

LOGGER_NAME = "cad_quoter"


def get_logger(*names: str) -> logging.Logger:
    """Return a logger under the shared CAD Quoter namespace."""
    if not names:
        return logging.getLogger(LOGGER_NAME)
    qualified = ".".join((LOGGER_NAME, *names))
    return logging.getLogger(qualified)


logger = get_logger()


def configure_logging(level: int = logging.INFO, *, force: bool = False) -> None:
    """Initialise a basic logging configuration if none is present."""
    root = logging.getLogger()
    if root.handlers and not force:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
