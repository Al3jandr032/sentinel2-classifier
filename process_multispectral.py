#!/usr/bin/env python3
"""Process full Sentinel-2 multispectral SAFE folder."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.sentinel2_classifier import (
    Sentinel2Classifier,
    create_sample_labels_from_index,
    load_sentinel2_multispectral,
    prepare_features,
    save_classified_raster,
    visualize_classification,
)


def main():
    # Path to Sentinel-2 SAFE folder
    safe_folder = "S2C_MSIL2A_20250813T165911_N0511_R069_T14QMG_20250813T231612.SAFE"  # Replace with actual path

    # Processing parameters
    target_resolution = 10  # 10m, 20m, or 60m
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

    try:
        print("Loading and resampling Sentinel-2 multispectral data...")
        data, profile, band_order = load_sentinel2_multispectral(
            safe_folder, target_resolution, selected_bands
        )

        print(f"Resampled data shape: {data.shape}")
        print(f"Band order: {band_order}")
        print(f"Target resolution: {target_resolution}m")

        # Prepare features for sklearn
        features = prepare_features(data)
        print(f"Features shape: {features.shape}")

        # Generate labels from indices
        labels = create_sample_labels_from_index(data, band_order)
        print(f"Generated {len(np.unique(labels))} classes")

        # Train classifier
        classifier = Sentinel2Classifier(
            RandomForestClassifier(n_estimators=100, random_state=42)
        )

        print("Training model...")
        classifier.train(features, labels)

        # Classify full image
        print("Classifying image...")
        predictions = classifier.predict(features)

        # Save results
        _, height, width = data.shape
        classified_image = predictions.reshape(height, width)

        # Save model and results
        classifier.save_model("multispectral_model.pkl")
        save_classified_raster(
            predictions, profile, "multispectral_classified.tif", height, width
        )
        visualize_classification(classified_image, "multispectral_map.png")

        print("Processing completed!")
        print("Model saved: multispectral_model.pkl")
        print("Classified raster: multispectral_classified.tif")
        print("Visualization: multispectral_map.png")

    except FileNotFoundError:
        print("Please provide a valid Sentinel-2 SAFE folder path")
        print("Example structure: S2A_MSIL2A_20231201T103421_*.SAFE")


if __name__ == "__main__":
    main()
