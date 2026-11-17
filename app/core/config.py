from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "anomatrix"
    environment: str = "development"
    debug: bool = True
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    dataset_path: str = "data/raw/telemetry_events.csv"
    model_artifact_path: str = "artifacts/models/autoencoder.pt"
    scaler_path: str = "artifacts/scalers/minmax_scaler.pkl"
    model_version: str = "v1.0.0"
    anomaly_threshold: float = 3.0
    random_seed: int = 42

    database_url: str | None = None
    log_level: str = "INFO"
    kafka_enabled: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "telemetry-events"
    kafka_consumer_group: str = "anomatrix-consumers"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=(),
    )


settings = Settings()
