from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TelemetryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str | None = Field(default=None, min_length=1, max_length=128)
    timestamp: datetime | None = None
    cpu: float = Field(ge=0, le=100, allow_inf_nan=False)
    memory: float = Field(ge=0, le=100, allow_inf_nan=False)
    latency: float = Field(ge=0, allow_inf_nan=False)
    network_traffic: float = Field(ge=0, allow_inf_nan=False)
    error_rate: float = Field(ge=0, allow_inf_nan=False)


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[TelemetryRequest] = Field(min_length=1, max_length=1000)
