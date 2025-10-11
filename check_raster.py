#!/usr/bin/env python3
"""Check Sentinel-2 or any raster image information."""

import sys

from src.sentinel2_classifier.raster_info import print_raster_info


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_raster.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        print_raster_info(image_path)
    except Exception as e:
        print(f"Error reading raster: {e}")


if __name__ == "__main__":
    main()
