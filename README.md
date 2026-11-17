# Anomatrix

Anomatrix is a FastAPI service that scores telemetry events with a saved PyTorch
autoencoder and Isolation Forest. It stores events and predictions in SQLite for
local use; a MySQL URL can be supplied for other deployments.

## Requirements

- Python 3.11
- The saved model files under `artifacts/` (included in the project)

## Run locally

In PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (!(Test-Path .env)) { Copy-Item .env.example .env }
alembic upgrade head
uvicorn app.main:app --reload
```

The local API is available at <http://127.0.0.1:8000>; interactive API docs are
at <http://127.0.0.1:8000/docs>. The dashboard is at <http://127.0.0.1:8000/>.
The default SQLite database is
`data/anomatrix.db`.

To regenerate the synthetic training dataset and model artifacts:

```powershell
python scripts/generate_synthetic_data.py
python scripts/train_model.py
```

Run the test suite with:

```powershell
python -m pytest
```

## Run with Docker Compose

Install Docker Desktop, then from the project root:

```powershell
docker compose up --build
```

Compose runs database migrations before starting the API and stores the SQLite
database in the persistent `anomatrix-data` volume. The service binds to
`127.0.0.1:8000` and exposes `/health`, `/ready`, `/model/info`, `/predict`, and
`/predict/batch`. Stop it with `Ctrl+C`; run `docker compose down` to remove the
container. The database volume is retained unless explicitly removed.

The API does not provide authentication. Keep the default loopback binding;
do not expose it directly to the internet without adding authentication and an
appropriately secured reverse proxy.

## Optional local Kafka demo

Kafka streaming is opt-in and runs only on your computer. With Docker Desktop
running, start the API, a local Kafka broker, and the prediction worker:

```powershell
docker compose -f compose.yaml -f compose.kafka.yaml up --build
```

Open <http://127.0.0.1:8000/> and choose **Send through Kafka**. The API queues
the event, the worker consumes it and saves the prediction, and the dashboard
shows the result. Stop the services with `Ctrl+C` or
`docker compose -f compose.yaml -f compose.kafka.yaml down`.

This mode uses no hosted broker or cloud Kafka account, so it has no Kafka
hosting or usage charge. It requires Docker Desktop and runs only while your
computer and Docker are on. The public Render demo remains REST-only and does
not depend on Kafka.

## Free public demo

`render.yaml` defines a public free-tier Render demo. The free instance may sleep
after inactivity and take time to wake up; it does not guarantee uninterrupted
24/7 availability. Its SQLite database is ephemeral, so saved predictions may
not persist across restarts or redeploys. The demo has no login, so do not send
real or sensitive telemetry.

To publish it, create a GitHub repository for this project (keep `.env` private
and out of Git), push the code, then create a new Blueprint on Render and
connect the repository. Render reads `render.yaml` to build and start the app.
After deployment, the dashboard is served at the service URL and API docs are
available at `/docs`.

## Project layout

- `app/`: API, configuration, database models/repository, schemas, inference,
  optional Kafka producer/worker, and model training/evaluation code
- `artifacts/`: model weights, scaler, baseline model, metadata, and evaluation
  report required by inference
- `data/`: local synthetic input data and SQLite database
- `migrations/`: Alembic database schema migrations
- `scripts/`: data generation, model training, and evaluation commands
- `tests/unit/`: API and database tests
