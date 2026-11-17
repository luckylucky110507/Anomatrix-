from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import torch

from app.core.config import settings
from app.ml.baseline import IsolationForestBaseline
from app.ml.evaluator import evaluate_predictions
from app.ml.preprocessing import FEATURE_COLUMNS, build_features_and_labels, load_telemetry_dataset, split_temporal_data, transform_features
from app.ml.trainer import AutoencoderTrainer


def main() -> None:
    dataset = load_telemetry_dataset(settings.dataset_path)
    _, _, test_df = split_temporal_data(dataset, train_ratio=0.7, val_ratio=0.15)
    X_test, y_test = build_features_and_labels(test_df)

    scaler = joblib.load(settings.scaler_path)
    X_test_scaled = transform_features(X_test, scaler)

    trainer = AutoencoderTrainer.load(settings.model_artifact_path)
    trainer.scaler = scaler
    autoencoder_scores = trainer.predict_scores(X_test)
    autoencoder_pred = trainer.predict_labels(X_test)

    baseline_model = IsolationForestBaseline.load(str(Path(settings.model_artifact_path).parent / "isolation_forest_baseline.joblib"))
    baseline_pred = baseline_model.predict(X_test_scaled)
    baseline_scores = -baseline_model.score_samples(X_test_scaled)

    auto_report = evaluate_predictions(y_test, autoencoder_pred, autoencoder_scores)
    baseline_report = evaluate_predictions(y_test, baseline_pred, baseline_scores)

    summary = {
        "autoencoder": auto_report,
        "isolation_forest": baseline_report,
        "dataset_rows": int(len(test_df)),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
