import rasterio

from .logging_config import get_logger

logger = get_logger(__name__)


def get_raster_info(image_path: str) -> dict:
    """Get comprehensive raster information."""
    with rasterio.open(image_path) as src:
        return {
            "bands": src.count,
            "width": src.width,
            "height": src.height,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs),
            "transform": src.transform,
            "bounds": src.bounds,
            "nodata": src.nodata,
        }


def print_raster_info(image_path: str) -> None:
    """Print formatted raster information."""
    info = get_raster_info(image_path)
    logger.info(f"File: {image_path}")
    logger.info(f"Bands: {info['bands']}")
    logger.info(f"Size: {info['width']} x {info['height']}")
    logger.info(f"Data type: {info['dtype']}")
    logger.info(f"CRS: {info['crs']}")
    logger.info(f"Bounds: {info['bounds']}")
    logger.info(f"NoData: {info['nodata']}")
