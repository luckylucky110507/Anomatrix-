from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from app.core.config import settings
from app.ml.autoencoder import Autoencoder, reconstruction_loss, set_seed
from app.ml.preprocessing import FEATURE_COLUMNS, build_features_and_labels, fit_scaler, transform_features


class TrainingConfig:
    def __init__(self, seed: int = 42, batch_size: int = 128, epochs: int = 30, learning_rate: float = 1e-3):
        self.seed = seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate


class AutoencoderTrainer:
    def __init__(self, feature_dim: int = len(FEATURE_COLUMNS), config: TrainingConfig | None = None):
        self.feature_dim = feature_dim
        self.config = config or TrainingConfig(seed=settings.random_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder(input_dim=self.feature_dim)
        self.model.to(self.device)
        self.scaler = None
        self.threshold = None
        self.history: list[dict[str, float]] = []

    def train(self, X_train: np.ndarray, X_val: np.ndarray) -> dict[str, Any]:
        set_seed(self.config.seed)
        self.scaler = fit_scaler(X_train, settings.scaler_path)
        X_train_scaled = transform_features(X_train, self.scaler)
        X_val_scaled = transform_features(X_val, self.scaler)

        train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = lambda pred, target: torch.mean(torch.pow(pred - target, 2), dim=1).mean()

        for epoch in range(self.config.epochs):
            self.model.train()
            permutation = torch.randperm(train_tensor.size(0))
            epoch_loss = 0.0

            for start in range(0, train_tensor.size(0), self.config.batch_size):
                idx = permutation[start:start + self.config.batch_size]
                batch = train_tensor[idx].to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)

            self.model.eval()
            with torch.no_grad():
                val_recon = self.model(val_tensor.to(self.device))
                val_loss = reconstruction_loss(val_tensor.to(self.device), val_recon).mean().item()

            self.history.append({"epoch": epoch + 1, "train_loss": epoch_loss / len(train_tensor), "val_loss": val_loss})

        self.threshold = self._select_threshold(val_tensor, self.model, self.device)
        return {"history": self.history, "threshold": self.threshold}

    def _select_threshold(self, val_tensor: torch.Tensor, model: Autoencoder, device: torch.device) -> float:
        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor.to(device))
            loss_values = reconstruction_loss(val_tensor.to(device), val_recon).cpu().numpy()

        # Use validation-based threshold: 99th percentile of validation loss to reduce false positives.
        threshold = float(np.percentile(loss_values, 99))
        return max(threshold, 1e-6)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("Model must be trained before prediction.")
        X_scaled = transform_features(X, self.scaler)
        tensor = torch.tensor(X_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(tensor.to(self.device))
            scores = reconstruction_loss(tensor.to(self.device), recon).cpu().numpy()
        return scores.astype(np.float32)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(X)
        return (scores > self.threshold).astype(int)

    def save(self, model_path: str | Path, metadata_path: str | Path) -> None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "feature_dim": self.feature_dim,
            "threshold": self.threshold,
            "config": {
                "seed": self.config.seed,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
            },
        }, model_path)

        metadata = {
            "feature_columns": FEATURE_COLUMNS,
            "threshold": float(self.threshold) if self.threshold is not None else None,
            "device": str(self.device),
            "history": self.history,
        }
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, model_path: str | Path, device: str | None = None) -> "AutoencoderTrainer":
        checkpoint = torch.load(model_path, map_location=device or "cpu", weights_only=True)
        trainer = cls(feature_dim=checkpoint["feature_dim"])
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.threshold = checkpoint["threshold"]
        trainer.model.to(trainer.device)
        trainer.model.eval()
        return trainer
