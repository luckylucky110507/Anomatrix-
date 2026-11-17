from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class APIResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class PredictionResponse(APIResponse):
    event_id: str | None
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    prediction: Literal["normal", "anomaly"]
    threshold: float
    model_version: str
    isolation_forest_score: float
    isolation_forest_is_anomaly: bool


class BatchPredictionResponse(APIResponse):
    count: int
    predictions: list[PredictionResponse]


class ModelInfoResponse(APIResponse):
    model_version: str
    model_type: str
    baseline_type: str
    features: list[str]
    threshold: float
