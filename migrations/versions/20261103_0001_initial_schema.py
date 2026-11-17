"""Create initial Anomatrix database tables.

Revision ID: 20261103_0001
Revises:
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261103_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("version", sa.String(length=100), nullable=False),
        sa.Column("model_type", sa.String(length=100), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("artifact_path", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint(
            "model_name",
            "version",
            name="uq_model_versions_name_version",
        ),
    )
    op.create_table(
        "telemetry_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.String(length=128), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("received_at", sa.DateTime(), nullable=False),
        sa.Column("cpu", sa.Float(), nullable=False),
        sa.Column("memory", sa.Float(), nullable=False),
        sa.Column("latency", sa.Float(), nullable=False),
        sa.Column("network_traffic", sa.Float(), nullable=False),
        sa.Column("error_rate", sa.Float(), nullable=False),
        sa.UniqueConstraint("event_id", name="uq_telemetry_events_event_id"),
    )
    op.create_index(
        "ix_telemetry_events_timestamp",
        "telemetry_events",
        ["timestamp"],
    )
    op.create_table(
        "anomaly_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("telemetry_event_id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("baseline_model_version_id", sa.Integer(), nullable=True),
        sa.Column("anomaly_score", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("is_anomaly", sa.Boolean(), nullable=False),
        sa.Column("isolation_forest_score", sa.Float(), nullable=False),
        sa.Column("isolation_forest_is_anomaly", sa.Boolean(), nullable=False),
        sa.Column("predicted_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["telemetry_event_id"],
            ["telemetry_events.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.ForeignKeyConstraint(
            ["baseline_model_version_id"],
            ["model_versions.id"],
        ),
        sa.UniqueConstraint(
            "telemetry_event_id",
            name="uq_anomaly_results_telemetry_event_id",
        ),
    )
    op.create_index(
        "ix_anomaly_results_is_anomaly",
        "anomaly_results",
        ["is_anomaly"],
    )
    op.create_index(
        "ix_anomaly_results_predicted_at",
        "anomaly_results",
        ["predicted_at"],
    )
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("model_version_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("dataset_rows", sa.Integer(), nullable=False),
        sa.Column("drift_share", sa.Float(), nullable=True),
        sa.Column("report_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
    )
    op.create_index(
        "ix_drift_reports_created_at",
        "drift_reports",
        ["created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_drift_reports_created_at", table_name="drift_reports")
    op.drop_table("drift_reports")
    op.drop_index("ix_anomaly_results_predicted_at", table_name="anomaly_results")
    op.drop_index("ix_anomaly_results_is_anomaly", table_name="anomaly_results")
    op.drop_table("anomaly_results")
    op.drop_index("ix_telemetry_events_timestamp", table_name="telemetry_events")
    op.drop_table("telemetry_events")
    op.drop_table("model_versions")
