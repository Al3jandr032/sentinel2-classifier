from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

from .geospatial_utils import (
    crop_multispectral_data,
    load_geojson,
    validate_and_transform_crs,
)
from .logging_config import get_logger

logger = get_logger(__name__)


def resample_sentinel2_bands(
    band_paths: Dict[str, str],
    target_resolution: int = 10,
    geojson_path: Optional[str] = None,
) -> Tuple[np.ndarray, dict]:
    """Load bands at target resolution only (no resampling for now)."""

    # Sentinel-2 band resolutions
    band_resolutions = {
        "B01": 60,
        "B02": 10,
        "B03": 10,
        "B04": 10,
        "B05": 20,
        "B06": 20,
        "B07": 20,
        "B08": 10,
        "B8A": 20,
        "B09": 60,
        "B10": 60,
        "B11": 20,
        "B12": 20,
    }

    # Filter bands to only those at target resolution
    target_bands = {
        band: path
        for band, path in band_paths.items()
        if band_resolutions.get(band) == target_resolution
    }

    if not target_bands:
        raise ValueError(f"No bands found at {target_resolution}m resolution")

    # Get reference band for profile
    ref_band = next(iter(target_bands.keys()))

    with rasterio.open(target_bands[ref_band]) as ref_src:
        ref_profile = ref_src.profile

    bands_data = []

    for band_name in sorted(target_bands.keys()):
        with rasterio.open(target_bands[band_name]) as src:
            bands_data.append(src.read(1))

    # Stack bands
    stacked_data = np.stack(bands_data, axis=0)

    # Update profile
    output_profile = ref_profile.copy()
    output_profile.update({"count": len(bands_data), "dtype": stacked_data.dtype})

    # Apply GeoJSON cropping if provided
    if geojson_path:
        geojson = load_geojson(geojson_path)
        logger.debug(f"Loaded GeoJSON: {geojson}")
        geojson = validate_and_transform_crs(geojson, str(ref_profile["crs"]))
        logger.debug(f"Validated GeoJSON: {geojson}")
        stacked_data, output_profile = crop_multispectral_data(
            stacked_data, output_profile, geojson
        )

    return stacked_data, output_profile


def load_sentinel2_safe_folder(
    safe_folder: str,
    target_resolution: int = 10,
    selected_bands: List[str] = None,
    geojson_path: Optional[str] = None,
) -> Tuple[np.ndarray, dict]:
    """Load Sentinel-2 SAFE folder, use only bands at target resolution."""
    from pathlib import Path

    # Define bands available at each resolution
    resolution_bands = {
        10: ["B02", "B03", "B04", "B08"],
        20: ["B05", "B06", "B07", "B8A", "B11", "B12"],
        60: ["B01", "B09", "B10"],
    }

    if selected_bands is None:
        selected_bands = resolution_bands[target_resolution]
    else:
        # Filter selected bands to only those available at target resolution
        available_bands = resolution_bands[target_resolution]
        selected_bands = [band for band in selected_bands if band in available_bands]

    # Find band files in SAFE folder
    safe_path = Path(safe_folder)
    img_folder = (
        safe_path / "GRANULE" / next(safe_path.glob("GRANULE/*")).name / "IMG_DATA"
    )

    band_paths = {}

    # Handle L1C vs L2A structure
    if (img_folder / "R10m").exists():  # L2A
        res_folder = f"R{target_resolution}m"
        res_path = img_folder / res_folder

        if res_path.exists():
            for band_file in res_path.glob("*.jp2"):
                for band in selected_bands:
                    if f"_{band}_" in band_file.name:
                        band_paths[band] = str(band_file)
                        break
    else:  # L1C
        for band_file in img_folder.glob("*.jp2"):
            for band in selected_bands:
                if f"_{band}_" in band_file.name:
                    band_paths[band] = str(band_file)
                    break

    return resample_sentinel2_bands(band_paths, target_resolution, geojson_path)


def create_common_resolution_dataset(
    band_data: np.ndarray, target_bands: List[int] = None
) -> np.ndarray:
    """Create sklearn-ready dataset from resampled bands."""
    if target_bands is not None:
        band_data = band_data[target_bands]

    bands, height, width = band_data.shape
    return band_data.reshape(bands, -1).T
