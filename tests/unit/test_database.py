from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import AnomalyResult, DriftReport, ModelVersion, TelemetryEvent
from app.db.repository import save_predictions
from app.schemas.prediction import PredictionResponse
from app.schemas.telemetry import TelemetryRequest


def make_test_session() -> tuple[Session, object]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return Session(engine), engine


def test_save_prediction_creates_related_rows_and_is_idempotent():
    session, engine = make_test_session()
    event = TelemetryRequest(
        event_id="db-test-event",
        cpu=45,
        memory=60,
        latency=120,
        network_traffic=300,
        error_rate=0.5,
    )
    prediction = PredictionResponse(
        event_id="db-test-event",
        timestamp=datetime.now(timezone.utc),
        anomaly_score=0.01,
        is_anomaly=False,
        prediction="normal",
        threshold=0.05,
        model_version="test-v1",
        isolation_forest_score=0.2,
        isolation_forest_is_anomaly=False,
    )

    first_saved = save_predictions(session, [event], [prediction])
    duplicate_saved = save_predictions(session, [event], [prediction])

    assert first_saved[0].event_id == duplicate_saved[0].event_id == "db-test-event"
    assert session.scalar(select(func.count()).select_from(TelemetryEvent)) == 1
    assert session.scalar(select(func.count()).select_from(AnomalyResult)) == 1
    assert session.scalar(select(func.count()).select_from(ModelVersion)) == 2

    session.close()
    engine.dispose()


def test_database_tables_are_registered():
    assert {
        "telemetry_events",
        "anomaly_results",
        "model_versions",
        "drift_reports",
    }.issubset(Base.metadata.tables)
    assert DriftReport.__tablename__ == "drift_reports"


def test_alembic_initial_migration_creates_all_tables(tmp_path: Path):
    database_path = tmp_path / "migration-test.sqlite"
    config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    try:
        table_names = set(engine.dialect.get_table_names(engine.connect()))
    finally:
        engine.dispose()
    assert {
        "alembic_version",
        "telemetry_events",
        "anomaly_results",
        "model_versions",
        "drift_reports",
    }.issubset(table_names)
