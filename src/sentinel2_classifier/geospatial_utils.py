import json
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds

from .logging_config import get_logger

logger = get_logger(__name__)


def load_geojson(geojson_path: str) -> dict:
    """Load GeoJSON file."""
    with open(geojson_path, "r") as f:
        return json.load(f)


def validate_and_transform_crs(geojson: dict, target_crs: str = "EPSG:4326") -> dict:
    """Validate and transform GeoJSON CRS to target CRS if needed."""
    # Check if CRS is specified
    crs = geojson.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    logger.debug(f"GeoJSON CRS: {crs}, target CRS: {target_crs}")

    if crs != target_crs:
        logger.info("Transforming from Web Mercator to WGS84")
        # Transform from Web Mercator to WGS84
        from pyproj import Transformer

        transformer = Transformer.from_crs(crs, target_crs, always_xy=True)

        # Transform coordinates
        for feature in geojson["features"]:
            coords = feature["geometry"]["coordinates"][0]  # Assuming polygon
            transformed_coords = []
            for coord in coords:
                x, y = transformer.transform(coord[0], coord[1])
                transformed_coords.append([x, y])
            feature["geometry"]["coordinates"] = [transformed_coords]

        # Update CRS
        geojson["crs"] = {"type": "name", "properties": {"name": target_crs}}

    return geojson


def crop_raster_with_geojson(
    raster_path: str, geojson: dict
) -> Tuple[np.ndarray, dict]:
    """Crop raster using GeoJSON polygon."""
    with rasterio.open(raster_path) as src:
        # Extract geometry from first feature
        geometry = geojson["features"][0]["geometry"]
        logger.debug(f"GeoJSON geometry: {geometry}")
        logger.debug(f"Raster CRS: {str(src.crs)}")
        logger.debug(f"Raster bounds: {src.bounds}")
        # Crop raster
        cropped_data, cropped_transform = mask(src, [geometry], crop=True, nodata=0)

        # Update profile
        profile = src.profile.copy()
        profile.update(
            {
                "height": cropped_data.shape[1],
                "width": cropped_data.shape[2],
                "transform": cropped_transform,
            }
        )

        return cropped_data, profile


def crop_multispectral_data(
    data: np.ndarray, profile: dict, geojson: dict
) -> Tuple[np.ndarray, dict]:
    """Crop multispectral data array using GeoJSON polygon."""
    # Create temporary raster to use rasterio mask
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
        temp_path = tmp_file.name

    try:
        # Write data to temporary file
        with rasterio.open(temp_path, "w", **profile) as dst:
            dst.write(data)

        # Crop using the temporary file
        cropped_data, cropped_profile = crop_raster_with_geojson(temp_path, geojson)

        return cropped_data, cropped_profile

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def get_roi_bounds(geojson: dict) -> Tuple[float, float, float, float]:
    """Get bounding box from GeoJSON polygon."""
    geometry = geojson["features"][0]["geometry"]
    coords = geometry["coordinates"][0]

    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]

    return min(lons), min(lats), max(lons), max(lats)  # minx, miny, maxx, maxy
