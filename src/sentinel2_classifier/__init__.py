"""Sentinel-2 Image Classification Demo Package."""

from .classifier import Sentinel2Classifier
from .data_loader import (
    create_sample_labels,
    create_sample_labels_from_index,
    load_sentinel2_image,
    load_sentinel2_multispectral,
    prepare_features,
)
from .indices import calculate_indices_from_sentinel2, calculate_ndvi, calculate_ndwi
from .raster_info import get_raster_info, print_raster_info
from .raster_processor import save_classified_raster, visualize_classification
from .resampling import (
    create_common_resolution_dataset,
    load_sentinel2_safe_folder,
    resample_sentinel2_bands,
)

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
]
