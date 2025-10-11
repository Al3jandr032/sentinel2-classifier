import numpy as np
import rasterio


def save_classified_raster(
    predictions: np.ndarray,
    original_profile: dict,
    output_path: str,
    height: int,
    width: int,
) -> None:
    """Save classification results as a GeoTIFF raster."""
    classified_image = predictions.reshape(height, width)

    profile = original_profile.copy()
    profile.update({"dtype": "uint8", "count": 1, "compress": "lzw"})

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(classified_image.astype("uint8"), 1)


def visualize_classification(
    classified_image: np.ndarray, output_path: str = None
) -> None:
    """Create a simple visualization of the classification."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(classified_image, cmap="viridis")
    plt.colorbar(label="Land Cover Class")
    plt.title("Sentinel-2 Classification Results")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
