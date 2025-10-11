"""Sentinel-2 Image Classification Demo Package."""

import os

from .classifier import Sentinel2Classifier
from .data_loader import (
    create_sample_labels,
    create_sample_labels_from_index,
    load_sentinel2_image,
    load_sentinel2_multispectral,
    prepare_features,
)
from .geospatial_utils import (
    crop_multispectral_data,
    get_roi_bounds,
    load_geojson,
    validate_and_transform_crs,
)
from .indices import calculate_indices_from_sentinel2, calculate_ndvi, calculate_ndwi
from .logging_config import get_logger, setup_logger
from .raster_info import get_raster_info, print_raster_info
from .raster_processor import save_classified_raster, visualize_classification
from .resampling import (
    create_common_resolution_dataset,
    load_sentinel2_safe_folder,
    resample_sentinel2_bands,
)

# Setup default logger
_log_level = os.getenv("SENTINEL2_LOG_LEVEL", "INFO")
setup_logger(level=_log_level)

__version__ = "0.1.0"
__all__ = [
    "load_sentinel2_image",
    "load_sentinel2_multispectral",
    "prepare_features",
    "create_sample_labels",
    "create_sample_labels_from_index",
    "Sentinel2Classifier",
    "save_classified_raster",
    "visualize_classification",
    "get_raster_info",
    "print_raster_info",
    "calculate_ndvi",
    "calculate_ndwi",
    "calculate_indices_from_sentinel2",
    "resample_sentinel2_bands",
    "load_sentinel2_safe_folder",
    "create_common_resolution_dataset",
    "load_geojson",
    "validate_and_transform_crs",
    "crop_multispectral_data",
    "get_roi_bounds",
    "get_logger",
    "setup_logger",
]
