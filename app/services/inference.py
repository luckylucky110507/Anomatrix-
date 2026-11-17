from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from app.core.config import Settings, settings
from app.ml.baseline import IsolationForestBaseline
from app.ml.preprocessing import FEATURE_COLUMNS, transform_features
from app.ml.trainer import AutoencoderTrainer
from app.schemas.prediction import PredictionResponse
from app.schemas.telemetry import TelemetryRequest


class InferenceService:
    """Loads the existing artifacts once and reuses them for each API request."""

    def __init__(
        self,
        autoencoder: AutoencoderTrainer,
        baseline: IsolationForestBaseline,
        scaler: object,
        model_version: str,
    ) -> None:
        if autoencoder.threshold is None:
            raise ValueError("The saved Autoencoder checkpoint does not contain a threshold.")
        self.autoencoder = autoencoder
        self.baseline = baseline
        self.scaler = scaler
        self.model_version = model_version
        self.threshold = float(autoencoder.threshold)

    @classmethod
    def from_artifacts(cls, config: Settings = settings) -> InferenceService:
        model_path = Path(config.model_artifact_path)
        scaler_path = Path(config.scaler_path)
        baseline_path = model_path.parent / "isolation_forest_baseline.joblib"

        autoencoder = AutoencoderTrainer.load(model_path)
        scaler = joblib.load(scaler_path)
        baseline = IsolationForestBaseline.load(str(baseline_path))

        if getattr(scaler, "n_features_in_", None) != len(FEATURE_COLUMNS):
            raise ValueError("The saved scaler does not match the telemetry feature count.")

        autoencoder.scaler = scaler
        return cls(
            autoencoder=autoencoder,
            baseline=baseline,
            scaler=scaler,
            model_version=config.model_version,
        )

    def predict(self, events: list[TelemetryRequest]) -> list[PredictionResponse]:
        if not events:
            raise ValueError("At least one telemetry event is required.")

        feature_matrix = np.asarray(
            [[getattr(event, feature) for feature in FEATURE_COLUMNS] for event in events],
            dtype=np.float32,
        )

        # The Autoencoder uses the scaler attached at load time; the baseline
        # receives values transformed by that same fitted scaler.
        autoencoder_scores = self.autoencoder.predict_scores(feature_matrix)
        autoencoder_predictions = self.autoencoder.predict_labels(feature_matrix)
        scaled_features = transform_features(feature_matrix, self.scaler)
        baseline_scores = -self.baseline.score_samples(scaled_features)
        baseline_predictions = self.baseline.predict(scaled_features)

        results: list[PredictionResponse] = []
        for index, event in enumerate(events):
            is_anomaly = bool(autoencoder_predictions[index])
            timestamp = event.timestamp or datetime.now(timezone.utc)
            results.append(
                PredictionResponse(
                    event_id=event.event_id,
                    timestamp=timestamp,
                    anomaly_score=float(autoencoder_scores[index]),
                    is_anomaly=is_anomaly,
                    prediction="anomaly" if is_anomaly else "normal",
                    threshold=self.threshold,
                    model_version=self.model_version,
                    isolation_forest_score=float(baseline_scores[index]),
                    isolation_forest_is_anomaly=bool(baseline_predictions[index]),
                )
            )
        return results
