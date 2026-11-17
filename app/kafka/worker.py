from __future__ import annotations

import logging
from collections.abc import Callable

from confluent_kafka import Consumer, KafkaError, KafkaException
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.repository import save_predictions
from app.db.session import get_session_factory
from app.schemas.telemetry import TelemetryRequest
from app.services.inference import InferenceService


logger = logging.getLogger("anomatrix.kafka.worker")


def process_telemetry_message(
    message_value: bytes,
    inference_service: InferenceService,
    session_factory: Callable[[], Session] | None = None,
) -> str:
    event = TelemetryRequest.model_validate_json(message_value)
    prediction = inference_service.predict([event])[0]
    if session_factory is None:
        session_factory = get_session_factory()
    with session_factory() as session:
        saved_prediction = save_predictions(session, [event], [prediction])[0]
    if saved_prediction.event_id is None:
        raise RuntimeError("The saved Kafka prediction is missing its event_id.")
    return saved_prediction.event_id


def run_worker() -> None:
    inference_service = InferenceService.from_artifacts()
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": settings.kafka_consumer_group,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
    )
    try:
        consumer.subscribe([settings.kafka_topic])
        logger.info("Consuming telemetry from Kafka topic %s", settings.kafka_topic)
        while True:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error() is not None:
                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(message.error())
            if message.value() is None:
                raise ValueError("Kafka telemetry message has no payload.")
            try:
                event_id = process_telemetry_message(
                    message.value(),
                    inference_service,
                )
                consumer.commit(message=message, asynchronous=False)
                logger.info("Processed Kafka telemetry event %s", event_id)
            except (ValidationError, ValueError, RuntimeError, SQLAlchemyError, OSError):
                logger.exception(
                    "Kafka telemetry processing failed; the message was not committed."
                )
                raise
    finally:
        consumer.close()


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    run_worker()
