"""Logging utilities for the telecom travel demand model."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: str,
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name (typically __name__).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to log file.
        format_string: Optional custom format string.

    Returns:
        Configured logger instance.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # Clear any existing handlers to prevent duplicates when called multiple times
    if logger.handlers:
        logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ProgressLogger:
    """
    Progress logging utility for long-running operations.

    Example:
        >>> progress = ProgressLogger(logger, total=1000, desc="Processing users")
        >>> for user in users:
        ...     # process user
        ...     progress.update()
        >>> progress.close()
    """

    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        desc: str = "Progress",
        log_interval: int = 10,
    ):
        """
        Initialize progress logger.

        Args:
            logger: Logger instance.
            total: Total number of items.
            desc: Description of the operation.
            log_interval: Percentage interval for logging (e.g., 10 = log every 10%).
        """
        self.logger = logger
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0
        self.last_logged_pct = -1

    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        self.current += n

        if self.total > 0:
            pct = int(100 * self.current / self.total)
            if pct // self.log_interval > self.last_logged_pct // self.log_interval:
                self.logger.info(f"{self.desc}: {self.current}/{self.total} ({pct}%)")
                self.last_logged_pct = pct

    def close(self) -> None:
        """Log completion."""
        self.logger.info(f"{self.desc}: Complete ({self.current}/{self.total})")
