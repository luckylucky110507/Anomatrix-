from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone


def generate_normal_row(rng: np.random.Generator) -> dict[str, float]:
    return {
        "cpu": float(max(0.0, min(100.0, rng.normal(45.0, 8.0)))),
        "memory": float(max(0.0, min(100.0, rng.normal(60.0, 10.0)))),
        "latency": float(max(0.0, rng.normal(120.0, 25.0))),
        "network_traffic": float(max(0.0, rng.normal(300.0, 80.0))),
        "error_rate": float(max(0.0, rng.normal(0.5, 0.3))),
    }


def generate_anomaly_row(rng: np.random.Generator) -> dict[str, float]:
    anomaly_type = rng.integers(0, 5)
    if anomaly_type == 0:
        return {
            "cpu": float(rng.uniform(80.0, 100.0)),
            "memory": float(rng.uniform(55.0, 80.0)),
            "latency": float(rng.uniform(150.0, 300.0)),
            "network_traffic": float(rng.uniform(400.0, 900.0)),
            "error_rate": float(rng.uniform(1.0, 8.0)),
        }
    if anomaly_type == 1:
        return {
            "cpu": float(rng.uniform(35.0, 70.0)),
            "memory": float(rng.uniform(85.0, 99.0)),
            "latency": float(rng.uniform(200.0, 450.0)),
            "network_traffic": float(rng.uniform(320.0, 720.0)),
            "error_rate": float(rng.uniform(0.8, 3.5)),
        }
    if anomaly_type == 2:
        return {
            "cpu": float(rng.uniform(10.0, 30.0)),
            "memory": float(rng.uniform(45.0, 65.0)),
            "latency": float(rng.uniform(350.0, 950.0)),
            "network_traffic": float(rng.uniform(500.0, 1200.0)),
            "error_rate": float(rng.uniform(2.0, 9.0)),
        }
    if anomaly_type == 3:
        return {
            "cpu": float(rng.uniform(60.0, 95.0)),
            "memory": float(rng.uniform(60.0, 85.0)),
            "latency": float(rng.uniform(180.0, 380.0)),
            "network_traffic": float(rng.uniform(1300.0, 2400.0)),
            "error_rate": float(rng.uniform(4.0, 12.0)),
        }
    return {
        "cpu": float(rng.uniform(20.0, 65.0)),
        "memory": float(rng.uniform(40.0, 70.0)),
        "latency": float(rng.uniform(250.0, 600.0)),
        "network_traffic": float(rng.uniform(1100.0, 1800.0)),
        "error_rate": float(rng.uniform(3.0, 15.0)),
    }


def generate_dataset(
    rows: int = 5000,
    seed: int = 42,
    output_path: str = "data/raw/telemetry_events.csv",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_time = datetime.now(timezone.utc) - timedelta(minutes=rows // 12)

    records: list[dict[str, object]] = []
    for i in range(rows):
        if i % 20 == 0:
            sample = generate_anomaly_row(rng)
            is_anomaly = 1
        else:
            sample = generate_normal_row(rng)
            is_anomaly = 0

        timestamp = start_time + timedelta(seconds=i * 5)
        records.append(
            {
                "timestamp": timestamp.isoformat(),
                "event_id": f"evt_{i:05d}",
                "cpu": sample["cpu"],
                "memory": sample["memory"],
                "latency": sample["latency"],
                "network_traffic": sample["network_traffic"],
                "error_rate": sample["error_rate"],
                "is_anomaly": is_anomaly,
                "label": is_anomaly,
                "source": "synthetic",
            }
        )

    df = pd.DataFrame(records)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} rows and saved to {output_file}")
    print(f"Anomaly ratio: {df['label'].mean():.2%}")
    return df


if __name__ == "__main__":
    generate_dataset()
