from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


FEATURE_COLUMNS = ["cpu", "memory", "latency", "network_traffic", "error_rate"]
TARGET_COLUMN = "label"


def load_telemetry_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def split_temporal_data(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    df = df.sort_values("timestamp").reset_index(drop=True)
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def build_features_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.int64)
    return X, y


def fit_scaler(X_train: np.ndarray, scaler_path: str | Path | None = None) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    if scaler_path is not None:
        scaler_path = Path(scaler_path)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
    return scaler


def transform_features(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(X).astype(np.float32)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
