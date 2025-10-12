from typing import Optional, Tuple

import numpy as np
import rasterio

from .indices import calculate_indices_from_sentinel2
from .logging_config import get_logger
from .resampling import create_common_resolution_dataset, load_sentinel2_safe_folder

logger = get_logger(__name__)


def load_sentinel2_image(image_path: str) -> Tuple[np.ndarray, dict]:
    """Load Sentinel-2 image and return data array with metadata."""
    with rasterio.open(image_path) as src:
        data = src.read()
        profile = src.profile
    return data, profile


def load_sentinel2_multispectral(
    safe_folder: str,
    target_resolution: int = 10,
    selected_bands: list = None,
    geojson_path: Optional[str] = None,
) -> Tuple[np.ndarray, dict, list]:
    """Load and resample Sentinel-2 SAFE folder to common resolution, optionally crop with GeoJSON."""
    # Define bands available at each resolution
    resolution_bands = {
        10: ["AOT", "B02", "B03", "B04", "B08", "TCI", "WVP"],
        20: [
            "AOT",
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B11",
            "B12",
            "B8A",
            "SCL",
            "TCI",
            "WVP",
        ],
        60: [
            "AOT",
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B09",
            "B11",
            "B12",
            "B8A",
            "SCL",
            "TCI",
            "WVP",
        ],
    }

    if selected_bands is None:
        selected_bands = resolution_bands[target_resolution]

    data, profile = load_sentinel2_safe_folder(
        safe_folder, target_resolution, selected_bands, geojson_path
    )
    return data, profile, selected_bands


def prepare_features(data: np.ndarray) -> np.ndarray:
    """Reshape image data for sklearn (pixels x bands)."""
    return create_common_resolution_dataset(data)


def create_sample_labels(height: int, width: int) -> np.ndarray:
    """Create simple demo labels (water=0, vegetation=1, urban=2)."""
    labels = np.zeros((height, width), dtype=np.uint8)
    # Simple pattern for demo
    labels[: height // 3, :] = 0  # water
    labels[height // 3 : 2 * height // 3, :] = 1  # vegetation
    labels[2 * height // 3 :, :] = 2  # urban
    return labels.flatten()


def create_sample_labels_from_index(
    data: np.ndarray, band_order: list = None
) -> np.ndarray:
    """Create labels based on NDVI and NDWI indices."""
    ndvi, ndwi = calculate_indices_from_sentinel2(data, band_order)

    labels = np.zeros_like(ndvi, dtype=np.uint8)

    # Water: high NDWI (> 0.3)
    labels[ndwi > 0.3] = 0

    # Vegetation: high NDVI (> 0.4) and low NDWI
    labels[(ndvi > 0.4) & (ndwi <= 0.3)] = 1

    # Urban: low NDVI and low NDWI
    labels[(ndvi <= 0.4) & (ndwi <= 0.3)] = 2

    return labels.flatten()
