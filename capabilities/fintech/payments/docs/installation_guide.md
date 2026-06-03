# APG Digital Payments — Installation Guide

## Requirements

- Python 3.12+
- PostgreSQL 14+ (optional — in-memory store works without it)
- `uv` or `pip`

## Standalone installation

```bash
pip install apg-fintech-payments
# or
uv add apg-fintech-payments
```

## Within APG platform

The capability is already registered in `capabilities/fintech/payments/`. No separate installation needed. The platform auto-discovers it via the `apg.capabilities` entry point.

## Database setup (PostgreSQL)

```bash
# Create database
createdb apg_payments

# Run schema
psql apg_payments < capabilities/fintech/payments/database/schema.sql

# Set environment variable
export APG_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/apg_payments"
```

The capability falls back to an in-memory store if `APG_DATABASE_URL` is not set — useful for development and testing.

## Flask Blueprint (standalone server)

```python
# app.py
import os
from flask import Flask
from capabilities.fintech.payments.blueprint import create_blueprint, create_ui_blueprint

app = Flask(__name__)
app.register_blueprint(create_blueprint(db_url=os.environ.get("APG_DATABASE_URL")))
app.register_blueprint(create_ui_blueprint())

if __name__ == "__main__":
    app.run(port=8080)
```

```bash
python app.py
# or
flask --app app run --port 8080
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `APG_DATABASE_URL` | No | PostgreSQL async URL. Falls back to in-memory. |
| `MPESA_CONSUMER_KEY` | For live M-Pesa | Safaricom Daraja consumer key |
| `MPESA_CONSUMER_SECRET` | For live M-Pesa | Safaricom Daraja consumer secret |
| `MPESA_PASSKEY` | For live M-Pesa | Daraja STK Push passkey |
| `MPESA_SHORTCODE` | For live M-Pesa | Business short code |
| `MTN_SUBSCRIPTION_KEY` | For live MTN MoMo | MTN MoMo API subscription key |
| `AIRTEL_CLIENT_ID` | For live Airtel | Airtel Money client ID |
| `AIRTEL_CLIENT_SECRET` | For live Airtel | Airtel Money client secret |

## Running tests

```bash
# From repo root
python -m pytest capabilities/fintech/payments/tests/ -v

# With coverage
python -m pytest capabilities/fintech/payments/tests/ --cov=capabilities.fintech.payments -v

# Type check
uv run pyright capabilities/fintech/payments/
```

## APG CLI registration

```bash
apg capability install fintech_payments
apg capability enable fintech_payments --tenant my-org
```

## Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[full]"
ENV APG_DATABASE_URL=postgresql+asyncpg://user:pass@db/apg
CMD ["python", "-m", "capabilities.fintech.payments.app"]
```

```bash
docker build -t apg-payments .
docker run -p 8080:8080 -e APG_DATABASE_URL=... apg-payments
```
