#!/usr/bin/env python3
"""Train a Sentinel-2 classification model."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.sentinel2_classifier.classifier import Sentinel2Classifier
from src.sentinel2_classifier.data_loader import (
    create_sample_labels_from_index,
    load_sentinel2_image,
    prepare_features,
)


def main():
    # Load Sentinel-2 image
    image_path = "path/to/sentinel2_image.tif"  # Replace with actual path

    try:
        data, profile = load_sentinel2_image(image_path)
        print(f"Loaded image with shape: {data.shape}")

        # Prepare features
        features = prepare_features(data)

        # Create labels from NDVI/NDWI indices
        labels = create_sample_labels_from_index(data)
        print(f"Generated {len(np.unique(labels))} classes from indices")

        # Initialize classifier (easily switchable)
        classifier = Sentinel2Classifier(
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        # classifier = Sentinel2Classifier(SVC(kernel='rbf', random_state=42))  # Alternative

        # Train model
        print("Training model...")
        classifier.train(features, labels)

        # Save trained model
        classifier.save_model("trained_model.pkl")
        print("Model saved to trained_model.pkl")

    except FileNotFoundError:
        print("Please provide a valid Sentinel-2 image path in the script")
        print("Demo completed - model structure is ready")


if __name__ == "__main__":
    main()
