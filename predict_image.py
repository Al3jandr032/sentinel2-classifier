#!/usr/bin/env python3
"""Classify new Sentinel-2 images using trained model."""

from src.sentinel2_classifier import setup_logger
from src.sentinel2_classifier.classifier import Sentinel2Classifier
from src.sentinel2_classifier.data_loader import load_sentinel2_image, prepare_features
from src.sentinel2_classifier.raster_processor import (
    save_classified_raster,
    visualize_classification,
)

# Setup logging
logger = setup_logger("predict_image", level="INFO")


def main():
    # Paths
    model_path = "trained_model.pkl"
    input_image = "path/to/new_sentinel2_image.tif"  # Replace with actual path
    output_raster = "classified_output.tif"

    try:
        # Load trained model
        classifier = Sentinel2Classifier()
        classifier.load_model(model_path)
        logger.info("Model loaded successfully")

        # Load new image
        data, profile = load_sentinel2_image(input_image)
        logger.info(f"Processing image with shape: {data.shape}")

        # Prepare features
        features = prepare_features(data)

        # Classify
        logger.info("Classifying image...")
        predictions = classifier.predict(features)

        # Save results
        _, height, width = data.shape
        save_classified_raster(predictions, profile, output_raster, height, width)
        logger.info(f"Classification saved to {output_raster}")

        # Visualize
        classified_image = predictions.reshape(height, width)
        visualize_classification(classified_image, "classification_map.png")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure model is trained and input image path is correct")


if __name__ == "__main__":
    main()
