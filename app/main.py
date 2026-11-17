from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.repository import get_prediction_by_event_id, save_predictions
from app.db.session import DatabaseConfigurationError, get_session_factory
from app.kafka.producer import KafkaPublishError, publish_telemetry_event
from app.ml.preprocessing import FEATURE_COLUMNS
from app.schemas.prediction import (
    BatchPredictionResponse,
    KafkaStreamStatusResponse,
    KafkaSubmissionResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from app.schemas.telemetry import BatchPredictionRequest, TelemetryRequest
from app.services.inference import InferenceService


logger = logging.getLogger("anomatrix.api")
DASHBOARD_PATH = Path(__file__).parent / "static" / "index.html"


def get_db() -> Iterator[Session]:
    try:
        session = get_session_factory()()
    except DatabaseConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not configured. Set DATABASE_URL in the private .env file.",
        ) from exc
    try:
        yield session
    finally:
        session.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if app.state.inference_service is None:
        try:
            app.state.inference_service = InferenceService.from_artifacts()
        except (FileNotFoundError, OSError, EOFError, ImportError, KeyError, RuntimeError, ValueError):
            app.state.model_load_failed = True
            logger.exception("Could not load the saved inference artifacts.")
    yield


def create_app(inference_service: InferenceService | None = None) -> FastAPI:
    application = FastAPI(
        title="Anomatrix Inference API",
        description="Telemetry anomaly predictions using the saved Anomatrix models.",
        version=settings.model_version,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.state.inference_service = inference_service
    application.state.model_load_failed = False

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        errors = [
            {
                "loc": error["loc"],
                "msg": error["msg"],
                "type": error["type"],
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": errors},
        )

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(DASHBOARD_PATH, media_type="text/html")

    @application.get("/ready")
    def ready(
        request: Request,
        session: Session = Depends(get_db),
    ) -> dict[str, str]:
        if request.app.state.inference_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference artifacts are unavailable.",
            )
        try:
            session.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            logger.exception("Database readiness check failed.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database is unavailable.",
            ) from exc
        return {"status": "ready"}

    @application.get("/model/info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        service = _get_service(request)
        return ModelInfoResponse(
            model_version=service.model_version,
            model_type="PyTorch Autoencoder",
            baseline_type="scikit-learn Isolation Forest",
            features=FEATURE_COLUMNS,
            threshold=service.threshold,
        )

    @application.get("/kafka/status")
    def kafka_status() -> dict[str, bool]:
        return {"enabled": settings.kafka_enabled}

    @application.post(
        "/predict/stream",
        response_model=KafkaSubmissionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def predict_stream(payload: TelemetryRequest) -> KafkaSubmissionResponse:
        if not settings.kafka_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Kafka streaming is disabled for this deployment.",
            )

        event = payload
        if event.event_id is None:
            event = event.model_copy(update={"event_id": str(uuid4())})
        try:
            publish_telemetry_event(event)
        except KafkaPublishError as exc:
            logger.exception("Could not publish telemetry to Kafka.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Kafka is unavailable. Check the local Kafka demo setup.",
            ) from exc
        return KafkaSubmissionResponse(event_id=event.event_id, status="queued")

    @application.get(
        "/predict/stream/{event_id}",
        response_model=KafkaStreamStatusResponse,
    )
    def stream_prediction_status(
        event_id: str,
        session: Session = Depends(get_db),
    ) -> KafkaStreamStatusResponse:
        if not settings.kafka_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Kafka streaming is disabled for this deployment.",
            )
        try:
            prediction = get_prediction_by_event_id(session, event_id)
        except SQLAlchemyError as exc:
            logger.exception("Could not read Kafka prediction status.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction status is unavailable because the database is unavailable.",
            ) from exc
        if prediction is None:
            return KafkaStreamStatusResponse(event_id=event_id, status="queued")
        return KafkaStreamStatusResponse(
            event_id=event_id,
            status="completed",
            prediction=prediction,
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(
        payload: TelemetryRequest,
        request: Request,
        session: Session = Depends(get_db),
    ) -> PredictionResponse:
        service = _get_service(request)
        try:
            prediction = service.predict([payload])[0]
            return save_predictions(session, [payload], [prediction])[0]
        except SQLAlchemyError as exc:
            session.rollback()
            logger.exception("Could not save prediction to the database.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction could not be saved because the database is unavailable.",
            ) from exc
        except (RuntimeError, ValueError, OSError) as exc:
            logger.exception("Telemetry prediction failed.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed. Check application logs.",
            ) from exc

    @application.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(
        payload: BatchPredictionRequest,
        request: Request,
        session: Session = Depends(get_db),
    ) -> BatchPredictionResponse:
        service = _get_service(request)
        try:
            predictions = service.predict(payload.events)
            predictions = save_predictions(session, payload.events, predictions)
        except SQLAlchemyError as exc:
            session.rollback()
            logger.exception("Could not save batch predictions to the database.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Batch predictions could not be saved because the database is unavailable.",
            ) from exc
        except (RuntimeError, ValueError, OSError) as exc:
            logger.exception("Batch telemetry prediction failed.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch prediction failed. Check application logs.",
            ) from exc
        return BatchPredictionResponse(count=len(predictions), predictions=predictions)

    return application


def _get_service(request: Request) -> InferenceService:
    service = request.app.state.inference_service
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference artifacts are unavailable.",
        )
    return service


app = create_app()
