from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import AnomalyResult, TelemetryEvent
from app.main import create_app, get_db
from app.schemas.prediction import PredictionResponse


class FakeInferenceService:
    model_version = "test-v1"
    threshold = 0.05

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def predict(self, events):
        if self.fail:
            raise RuntimeError("private test error")
        results = []
        for event in events:
            results.append(
                PredictionResponse(
                    event_id=event.event_id,
                    timestamp=event.timestamp or datetime.now(timezone.utc),
                    anomaly_score=0.01,
                    is_anomaly=False,
                    prediction="normal",
                    threshold=self.threshold,
                    model_version=self.model_version,
                    isolation_forest_score=0.2,
                    isolation_forest_is_anomaly=False,
                )
            )
        return results


@pytest.fixture
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    test_session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_get_db():
        with test_session_factory() as session:
            yield session

    application = create_app(FakeInferenceService())
    application.dependency_overrides[get_db] = override_get_db
    application.state.test_session_factory = test_session_factory
    try:
        with TestClient(application) as test_client:
            yield test_client
    finally:
        engine.dispose()


def valid_telemetry(**overrides):
    event = {
        "event_id": "test-event-1",
        "cpu": 45,
        "memory": 60,
        "latency": 120,
        "network_traffic": 300,
        "error_rate": 0.5,
    }
    return {**event, **overrides}


def test_health_and_readiness(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Anomatrix — Telemetry intelligence" in response.text


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["model_version"] == "test-v1"
    assert response.json()["threshold"] == 0.05
    assert response.json()["features"] == [
        "cpu",
        "memory",
        "latency",
        "network_traffic",
        "error_rate",
    ]


def test_single_prediction(client):
    response = client.post("/predict", json=valid_telemetry())
    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == "test-event-1"
    assert body["prediction"] == "normal"
    assert body["is_anomaly"] is False
    assert body["threshold"] == 0.05
    assert body["model_version"] == "test-v1"
    assert "timestamp" in body
    with client.app.state.test_session_factory() as session:
        assert session.scalar(select(func.count()).select_from(TelemetryEvent)) == 1
        assert session.scalar(select(func.count()).select_from(AnomalyResult)) == 1


def test_batch_prediction(client):
    response = client.post(
        "/predict/batch",
        json={"events": [valid_telemetry(), valid_telemetry(event_id="test-event-2")]},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["predictions"]) == 2


def test_duplicate_event_is_not_stored_twice(client):
    payload = valid_telemetry()
    first = client.post("/predict", json=payload)
    second = client.post("/predict", json=payload)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    with client.app.state.test_session_factory() as session:
        assert session.scalar(select(func.count()).select_from(TelemetryEvent)) == 1
        assert session.scalar(select(func.count()).select_from(AnomalyResult)) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"cpu": -1},
        {"cpu": 101},
        {"memory": -0.1},
        {"latency": "NaN"},
        {"network_traffic": "Infinity"},
        {"unexpected_field": "not allowed"},
    ],
)
def test_invalid_telemetry_returns_validation_error(client, overrides):
    response = client.post("/predict", json=valid_telemetry(**overrides))
    assert response.status_code == 422


def test_empty_batch_is_rejected(client):
    response = client.post("/predict/batch", json={"events": []})
    assert response.status_code == 422


def test_non_finite_json_number_returns_validation_error(client):
    payload = (
        '{"event_id":"test-event-1","cpu":45,"memory":60,"latency":NaN,'
        '"network_traffic":300,"error_rate":0.5}'
    )
    response = client.post(
        "/predict",
        content=payload,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert "input" not in response.json()["detail"][0]


def test_unavailable_artifacts_make_app_not_ready():
    with TestClient(create_app()) as test_client:
        # Replacing the service after startup simulates an unavailable model set.
        test_client.app.state.inference_service = None
        assert test_client.get("/ready").status_code == 503
        assert test_client.post("/predict", json=valid_telemetry()).status_code == 503


def test_prediction_error_returns_safe_server_error(client):
    client.app.state.inference_service = FakeInferenceService(fail=True)
    response = client.post("/predict", json=valid_telemetry())
    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed. Check application logs."
