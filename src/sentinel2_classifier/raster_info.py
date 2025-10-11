import rasterio


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
    print(f"File: {image_path}")
    print(f"Bands: {info['bands']}")
    print(f"Size: {info['width']} x {info['height']}")
    print(f"Data type: {info['dtype']}")
    print(f"CRS: {info['crs']}")
    print(f"Bounds: {info['bounds']}")
    print(f"NoData: {info['nodata']}")
