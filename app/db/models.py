from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TelemetryEvent(Base):
    __tablename__ = "telemetry_events"
    __table_args__ = (
        Index("ix_telemetry_events_timestamp", "timestamp"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)
    cpu: Mapped[float] = mapped_column(Float, nullable=False)
    memory: Mapped[float] = mapped_column(Float, nullable=False)
    latency: Mapped[float] = mapped_column(Float, nullable=False)
    network_traffic: Mapped[float] = mapped_column(Float, nullable=False)
    error_rate: Mapped[float] = mapped_column(Float, nullable=False)

    anomaly_result: Mapped[AnomalyResult | None] = relationship(
        back_populates="telemetry_event",
        cascade="all, delete-orphan",
        uselist=False,
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_versions_name_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    anomaly_results: Mapped[list[AnomalyResult]] = relationship(
        back_populates="model_version",
        foreign_keys="AnomalyResult.model_version_id",
    )
    baseline_results: Mapped[list[AnomalyResult]] = relationship(
        back_populates="baseline_model_version",
        foreign_keys="AnomalyResult.baseline_model_version_id",
    )
    drift_reports: Mapped[list[DriftReport]] = relationship(back_populates="model_version")


class AnomalyResult(Base):
    __tablename__ = "anomaly_results"
    __table_args__ = (
        Index("ix_anomaly_results_is_anomaly", "is_anomaly"),
        Index("ix_anomaly_results_predicted_at", "predicted_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    telemetry_event_id: Mapped[int] = mapped_column(
        ForeignKey("telemetry_events.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"),
        nullable=False,
    )
    baseline_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"),
        nullable=True,
    )
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    isolation_forest_score: Mapped[float] = mapped_column(Float, nullable=False)
    isolation_forest_is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    telemetry_event: Mapped[TelemetryEvent] = relationship(back_populates="anomaly_result")
    model_version: Mapped[ModelVersion] = relationship(
        back_populates="anomaly_results",
        foreign_keys=[model_version_id],
    )
    baseline_model_version: Mapped[ModelVersion | None] = relationship(
        back_populates="baseline_results",
        foreign_keys=[baseline_model_version_id],
    )


class DriftReport(Base):
    __tablename__ = "drift_reports"
    __table_args__ = (
        Index("ix_drift_reports_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    dataset_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    drift_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    report_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, nullable=False)

    model_version: Mapped[ModelVersion | None] = relationship(back_populates="drift_reports")
