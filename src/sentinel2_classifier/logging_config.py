import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "sentinel2_classifier",
    level: str = "INFO",
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Setup logger with configurable level and format."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def get_logger(name: str = "sentinel2_classifier") -> logging.Logger:
    """Get existing logger or create with default settings."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
