from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from app.core.config import settings
from app.ml.autoencoder import Autoencoder, set_seed
from app.ml.baseline import IsolationForestBaseline
from app.ml.evaluator import evaluate_predictions, save_evaluation_report
from app.ml.preprocessing import FEATURE_COLUMNS, build_features_and_labels, load_telemetry_dataset, save_json, split_temporal_data, transform_features
from app.ml.trainer import AutoencoderTrainer


def main() -> None:
    data_path = settings.dataset_path
    dataset = load_telemetry_dataset(data_path)
    train_df, val_df, test_df = split_temporal_data(dataset, train_ratio=0.7, val_ratio=0.15)

    normal_train_df = train_df[train_df["label"] == 0].copy()
    normal_val_df = val_df[val_df["label"] == 0].copy()
    X_train, y_train = build_features_and_labels(normal_train_df)
    X_val, y_val = build_features_and_labels(normal_val_df)
    X_test, y_test = build_features_and_labels(test_df)

    set_seed(settings.random_seed)
    autoencoder_trainer = AutoencoderTrainer(feature_dim=len(FEATURE_COLUMNS))
    autoencoder_trainer.train(X_train, X_val)

    if autoencoder_trainer.scaler is None:
        raise RuntimeError("Scaler was not fitted during training.")

    X_train_scaled = transform_features(X_train, autoencoder_trainer.scaler)
    X_val_scaled = transform_features(X_val, autoencoder_trainer.scaler)
    X_test_scaled = transform_features(X_test, autoencoder_trainer.scaler)

    autoencoder_scores_test = autoencoder_trainer.predict_scores(X_test)
    autoencoder_pred_test = autoencoder_trainer.predict_labels(X_test)

    test_threshold = float(autoencoder_trainer.threshold)

    # Isolation Forest baseline
    baseline_model = IsolationForestBaseline(contamination=0.05, random_state=settings.random_seed)
    baseline_model.fit(X_train_scaled)
    baseline_pred = baseline_model.predict(X_test_scaled)
    baseline_scores = -baseline_model.score_samples(X_test_scaled)

    auto_report = evaluate_predictions(y_test, autoencoder_pred_test, autoencoder_scores_test)
    baseline_report = evaluate_predictions(y_test, baseline_pred, baseline_scores)

    # Save model artifacts and reports
    model_artifact_dir = Path(settings.model_artifact_path).parent
    model_artifact_dir.mkdir(parents=True, exist_ok=True)
    autoencoder_trainer.save(
        model_path=settings.model_artifact_path,
        metadata_path=model_artifact_dir / "autoencoder_metadata.json",
    )
    baseline_model.save(str(model_artifact_dir / "isolation_forest_baseline.joblib"))

    metric_summary = {
        "threshold": test_threshold,
        "autoencoder": auto_report,
        "isolation_forest": baseline_report,
        "normal_train_samples": int(len(normal_train_df)),
        "normal_val_samples": int(len(normal_val_df)),
        "val_slice_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
    }
    save_evaluation_report(model_artifact_dir / "evaluation_summary.json", metric_summary)

    print("=== Phase 3 — Model Training and Evaluation ===")
    print(f"Normal training samples: {len(normal_train_df)}")
    print(f"Normal validation samples: {len(normal_val_df)} (threshold calibration)")
    print(f"Validation slice total: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Autoencoder threshold: {test_threshold:.6f}")
    print("Autoencoder metrics:")
    print(json.dumps(auto_report, indent=2))
    print("Isolation Forest metrics:")
    print(json.dumps(baseline_report, indent=2))


if __name__ == "__main__":
    main()
