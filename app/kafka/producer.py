from __future__ import annotations

from confluent_kafka import KafkaException, Producer

from app.core.config import settings
from app.schemas.telemetry import TelemetryRequest


class KafkaPublishError(RuntimeError):
    """Raised when a telemetry event cannot be acknowledged by Kafka."""


def publish_telemetry_event(event: TelemetryRequest) -> None:
    if event.event_id is None:
        raise ValueError("A Kafka telemetry event must have an event_id.")

    delivery_errors: list[str] = []

    def on_delivery(error, message) -> None:
        if error is not None:
            delivery_errors.append(
                f"{message.topic()}[{message.partition()}]: {error}"
            )

    try:
        producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
        producer.produce(
            topic=settings.kafka_topic,
            key=event.event_id,
            value=event.model_dump_json(),
            on_delivery=on_delivery,
        )
        pending_messages = producer.flush(timeout=10.0)
    except (KafkaException, BufferError) as exc:
        raise KafkaPublishError("Could not publish telemetry to Kafka.") from exc

    if pending_messages:
        raise KafkaPublishError("Kafka did not acknowledge the event before timeout.")
    if delivery_errors:
        raise KafkaPublishError(delivery_errors[0])
