import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


def calculate_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """Calculate NDVI: (NIR - Red) / (NIR + Red)"""
    return np.divide(
        nir_band - red_band,
        nir_band + red_band,
        out=np.zeros_like(nir_band),
        where=(nir_band + red_band) != 0,
    )


def calculate_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """Calculate NDWI: (Green - NIR) / (Green + NIR)"""
    return np.divide(
        green_band - nir_band,
        green_band + nir_band,
        out=np.zeros_like(green_band),
        where=(green_band + nir_band) != 0,
    )


def calculate_indices_from_sentinel2(
    data: np.ndarray, band_order: list = None
) -> tuple:
    """Calculate NDVI and NDWI from Sentinel-2 data with flexible band order."""
    if band_order is None:
        # Default order for resampled bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
        band_order = [
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
        ]

    logger.debug(f"Calculating indices with band order: {band_order}")

    # Find band indices
    try:
        green_idx = band_order.index("B03")  # Green
        red_idx = band_order.index("B04")  # Red
        nir_idx = band_order.index("B08")  # NIR
        logger.debug(
            f"Band indices - Green: {green_idx}, Red: {red_idx}, NIR: {nir_idx}"
        )
    except ValueError:
        # Fallback to positional indexing if band names not found
        green_idx, red_idx, nir_idx = 1, 2, 3
        logger.warning("Band names not found, using positional indexing")

    green = data[green_idx].astype(np.float32)
    red = data[red_idx].astype(np.float32)
    nir = data[nir_idx].astype(np.float32)

    logger.info("Calculating NDVI and NDWI indices")
    ndvi = calculate_ndvi(red, nir)
    ndwi = calculate_ndwi(green, nir)

    return ndvi, ndwi
