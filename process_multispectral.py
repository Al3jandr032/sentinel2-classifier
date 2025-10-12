#!/usr/bin/env python3
"""Process full Sentinel-2 multispectral SAFE folder."""

import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.sentinel2_classifier import (
    Sentinel2Classifier,
    create_sample_labels_from_index,
    load_sentinel2_multispectral,
    prepare_features,
    save_classified_raster,
    setup_logger,
    visualize_classification,
)

# Setup logging
logger = setup_logger("process_multispectral", level="INFO")


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def main():
    # Load configuration
    config = load_config()
    safe_folder = config["safe_folder"]
    geoJson = config["geojson_path"]
    target_resolution = config["target_resolution"]
    selected_bands = config["selected_bands"]

    try:
        logger.info("Loading and resampling Sentinel-2 multispectral data...")
        data, profile, band_order = load_sentinel2_multispectral(
            safe_folder, target_resolution, selected_bands, geoJson
        )

        logger.info(f"Resampled data shape: {data.shape}")
        logger.info(f"Band order: {band_order}")
        logger.info(f"Target resolution: {target_resolution}m")

        # Prepare features for sklearn
        features = prepare_features(data)
        logger.info(f"Features shape: {features.shape}")

        # Generate labels from indices
        labels = create_sample_labels_from_index(data, band_order)
        logger.info(f"Generated {len(np.unique(labels))} classes")

        # Train classifier
        classifier = Sentinel2Classifier(
            RandomForestClassifier(n_estimators=100, random_state=42)
        )

        logger.info("Training model...")
        classifier.train(features, labels)

        # Classify full image
        logger.info("Classifying image...")
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

        logger.info("Processing completed!")
        logger.info("Model saved: multispectral_model.pkl")
        logger.info("Classified raster: multispectral_classified.tif")
        logger.info("Visualization: multispectral_map.png")

    except FileNotFoundError:
        logger.error("Please provide a valid Sentinel-2 SAFE folder path")
        logger.info("Example structure: S2A_MSIL2A_20231201T103421_*.SAFE")


if __name__ == "__main__":
    main()
