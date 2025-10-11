import pickle

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


class Sentinel2Classifier:
    """Wrapper for sklearn classifiers with easy model switching."""

    def __init__(self, classifier: BaseEstimator = None):
        self.classifier = classifier or RandomForestClassifier(
            n_estimators=50, random_state=42
        )

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the classifier."""
        self.classifier.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels for new features."""
        return self.classifier.predict(features)

    def save_model(self, filepath: str) -> None:
        """Save trained model to pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(self.classifier, f)

    def load_model(self, filepath: str) -> None:
        """Load trained model from pickle file."""
        with open(filepath, "rb") as f:
            self.classifier = pickle.load(f)
