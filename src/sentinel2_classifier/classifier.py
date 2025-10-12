import pickle

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from .logging_config import get_logger

logger = get_logger(__name__)


class Sentinel2Classifier:
    """Wrapper for sklearn classifiers with easy model switching."""

    def __init__(self, classifier: BaseEstimator = None):
        self.classifier = (
            classifier
            if classifier is not None
            else RandomForestClassifier(n_estimators=50, random_state=42)
        )

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the classifier."""
        logger.info(
            f"Training classifier with {features.shape[0]} samples and {features.shape[1]} features"
        )
        self.classifier.fit(features, labels)
        logger.info("Training completed")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels for new features."""
        logger.info(f"Predicting labels for {features.shape[0]} samples")
        predictions = self.classifier.predict(features)
        logger.debug(
            f"Prediction completed with {len(np.unique(predictions))} unique classes"
        )
        return predictions

    def save_model(self, filepath: str) -> None:
        """Save trained model to pickle file."""
        logger.info(f"Saving model to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(self.classifier, f)
        logger.info("Model saved successfully")

    def load_model(self, filepath: str) -> None:
        """Load trained model from pickle file."""
        logger.info(f"Loading model from {filepath}")
        with open(filepath, "rb") as f:
            self.classifier = pickle.load(f)
        logger.info("Model loaded successfully")
