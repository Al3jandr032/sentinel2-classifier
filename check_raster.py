#!/usr/bin/env python3
"""Check Sentinel-2 or any raster image information."""

import sys

from src.sentinel2_classifier import setup_logger
from src.sentinel2_classifier.raster_info import print_raster_info

# Setup logging
logger = setup_logger("check_raster", level="INFO")


def main():
    if len(sys.argv) != 2:
        logger.error("Usage: python check_raster.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        print_raster_info(image_path)
    except Exception as e:
        logger.error(f"Error reading raster: {e}")


if __name__ == "__main__":
    main()
