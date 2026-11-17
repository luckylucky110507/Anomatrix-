from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestBaseline:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(contamination=contamination, random_state=random_state)

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw_pred = self.model.predict(X)
        return (raw_pred == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.model.score_samples(X)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "IsolationForestBaseline":
        model = cls.__new__(cls)
        model.model = joblib.load(path)
        return model
