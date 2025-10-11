#!/usr/bin/env python3
"""Example script showing how to configure logging levels."""

import os

from src.sentinel2_classifier import setup_logger


# Example configurations
def demo_logging_levels():
    """Demonstrate different logging levels."""

    # Set environment variable to control logging level
    # os.environ["SENTINEL2_LOG_LEVEL"] = "DEBUG"

    # Setup logger with different levels
    logger_info = setup_logger("demo_info", level="INFO")
    logger_debug = setup_logger("demo_debug", level="DEBUG")

    logger_info.info("This is an INFO message")
    logger_info.debug("This DEBUG message won't show with INFO level")

    logger_debug.info("This is an INFO message from DEBUG logger")
    logger_debug.debug("This DEBUG message will show with DEBUG level")

    print("\nTo control logging globally, set environment variable:")
    print("export SENTINEL2_LOG_LEVEL=DEBUG")
    print("export SENTINEL2_LOG_LEVEL=INFO")
    print("export SENTINEL2_LOG_LEVEL=WARNING")
    print("export SENTINEL2_LOG_LEVEL=ERROR")


if __name__ == "__main__":
    demo_logging_levels()
