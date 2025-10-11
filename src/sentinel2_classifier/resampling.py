from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def resample_sentinel2_bands(
    band_paths: Dict[str, str], target_resolution: int = 10
) -> Tuple[np.ndarray, dict]:
    """Resample all Sentinel-2 bands to target resolution (10m, 20m, or 60m)."""

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

    # Get reference band (first band at target resolution)
    ref_band = next(
        band
        for band, res in band_resolutions.items()
        if res == target_resolution and band in band_paths
    )

    # Read reference for target shape and transform
    with rasterio.open(band_paths[ref_band]) as ref_src:
        ref_profile = ref_src.profile
        target_shape = (ref_src.height, ref_src.width)
        target_transform = ref_src.transform
        target_crs = ref_src.crs

    resampled_bands = []
    band_names = []

    for band_name in sorted(band_paths.keys()):
        with rasterio.open(band_paths[band_name]) as src:
            if band_resolutions[band_name] == target_resolution:
                # No resampling needed
                data = src.read(1)
            else:
                # Resample to target resolution
                data = np.empty(target_shape, dtype=src.dtypes[0])
                reproject(
                    source=rasterio.band(src, 1),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

            resampled_bands.append(data)
            band_names.append(band_name)

    # Stack bands
    stacked_data = np.stack(resampled_bands, axis=0)

    # Update profile
    output_profile = ref_profile.copy()
    output_profile.update({"count": len(resampled_bands), "dtype": stacked_data.dtype})

    return stacked_data, output_profile


def load_sentinel2_safe_folder(
    safe_folder: str, target_resolution: int = 10, selected_bands: List[str] = None
) -> Tuple[np.ndarray, dict]:
    """Load Sentinel-2 SAFE folder and resample bands to target resolution."""
    from pathlib import Path

    if selected_bands is None:
        selected_bands = [
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

    # Find band files in SAFE folder
    safe_path = Path(safe_folder)
    img_folder = (
        safe_path / "GRANULE" / next(safe_path.glob("GRANULE/*")).name / "IMG_DATA"
    )

    # Handle L1C vs L2A structure
    if (img_folder / "R10m").exists():  # L2A
        resolutions = {"R10m": 10, "R20m": 20, "R60m": 60}
        band_paths = {}

        for res_folder, resolution in resolutions.items():
            res_path = img_folder / res_folder
            if res_path.exists():
                for band_file in res_path.glob("*.jp2"):
                    band_name = None
                    for band in selected_bands:
                        if f"_{band}_" in band_file.name:
                            band_name = band
                            break
                    if band_name:
                        band_paths[band_name] = str(band_file)
    else:  # L1C
        band_paths = {}
        for band_file in img_folder.glob("*.jp2"):
            for band in selected_bands:
                if f"_{band}_" in band_file.name:
                    band_paths[band] = str(band_file)
                    break

    return resample_sentinel2_bands(band_paths, target_resolution)


def create_common_resolution_dataset(
    band_data: np.ndarray, target_bands: List[int] = None
) -> np.ndarray:
    """Create sklearn-ready dataset from resampled bands."""
    if target_bands is not None:
        band_data = band_data[target_bands]

    bands, height, width = band_data.shape
    return band_data.reshape(bands, -1).T
