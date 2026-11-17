from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.db.models import AnomalyResult, ModelVersion, TelemetryEvent
from app.schemas.prediction import PredictionResponse
from app.schemas.telemetry import TelemetryRequest


def _database_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _get_or_create_model_version(
    session: Session,
    model_name: str,
    version: str,
    model_type: str,
    threshold: float | None = None,
) -> ModelVersion:
    model_version = session.scalar(
        select(ModelVersion).where(
            ModelVersion.model_name == model_name,
            ModelVersion.version == version,
        )
    )
    if model_version is None:
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            model_type=model_type,
            threshold=threshold,
        )
        session.add(model_version)
        session.flush()
    return model_version


def _existing_prediction(
    session: Session,
    event_id: str,
) -> PredictionResponse | None:
    result = session.scalar(
        select(AnomalyResult)
        .join(AnomalyResult.telemetry_event)
        .options(
            joinedload(AnomalyResult.model_version),
            joinedload(AnomalyResult.baseline_model_version),
        )
        .where(TelemetryEvent.event_id == event_id)
    )
    if result is None:
        return None

    event = result.telemetry_event
    return PredictionResponse(
        event_id=event.event_id,
        timestamp=event.timestamp.replace(tzinfo=timezone.utc),
        anomaly_score=result.anomaly_score,
        is_anomaly=result.is_anomaly,
        prediction="anomaly" if result.is_anomaly else "normal",
        threshold=result.threshold,
        model_version=result.model_version.version,
        isolation_forest_score=result.isolation_forest_score,
        isolation_forest_is_anomaly=result.isolation_forest_is_anomaly,
    )


def save_predictions(
    session: Session,
    events: list[TelemetryRequest],
    predictions: list[PredictionResponse],
) -> list[PredictionResponse]:
    if len(events) != len(predictions):
        raise ValueError("The number of events and predictions must match.")

    saved_predictions: list[PredictionResponse] = []
    saved_by_event_id: dict[str, PredictionResponse] = {}
    with session.begin():
        for event, prediction in zip(events, predictions, strict=True):
            event_id = event.event_id or str(uuid4())
            if event_id in saved_by_event_id:
                saved_predictions.append(saved_by_event_id[event_id])
                continue
            existing = _existing_prediction(session, event_id)
            if existing is not None:
                saved_predictions.append(existing)
                saved_by_event_id[event_id] = existing
                continue

            telemetry_event = TelemetryEvent(
                event_id=event_id,
                timestamp=_database_datetime(
                    event.timestamp or prediction.timestamp
                ),
                cpu=event.cpu,
                memory=event.memory,
                latency=event.latency,
                network_traffic=event.network_traffic,
                error_rate=event.error_rate,
            )
            autoencoder_version = _get_or_create_model_version(
                session,
                model_name="autoencoder",
                version=prediction.model_version,
                model_type="PyTorch Autoencoder",
                threshold=prediction.threshold,
            )
            baseline_version = _get_or_create_model_version(
                session,
                model_name="isolation_forest",
                version=prediction.model_version,
                model_type="scikit-learn Isolation Forest",
            )

            session.add(telemetry_event)
            session.flush()
            result = AnomalyResult(
                telemetry_event_id=telemetry_event.id,
                model_version_id=autoencoder_version.id,
                baseline_model_version_id=baseline_version.id,
                anomaly_score=prediction.anomaly_score,
                threshold=prediction.threshold,
                is_anomaly=prediction.is_anomaly,
                isolation_forest_score=prediction.isolation_forest_score,
                isolation_forest_is_anomaly=prediction.isolation_forest_is_anomaly,
                predicted_at=_database_datetime(prediction.timestamp),
            )
            session.add(result)
            saved_prediction = prediction.model_copy(update={"event_id": event_id})
            saved_predictions.append(saved_prediction)
            saved_by_event_id[event_id] = saved_prediction

    return saved_predictions
