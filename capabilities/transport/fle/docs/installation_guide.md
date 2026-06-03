# Fleet Management — Installation Guide

## Requirements

- Python 3.12+
- PostgreSQL 14+ (for production)
- Flask 3.x
- Pydantic v2
- uuid6

## Install

```bash
pip install flask pydantic uuid6
# or with uv:
uv add flask pydantic uuid6
```

## Standalone Flask app

```python
from flask import Flask
from capabilities.transport.fle import register_capability

app = Flask(__name__)
register_capability(app)

if __name__ == "__main__":
    app.run(debug=True)
```

Access:
- API: `http://localhost:5000/api/fle/v1/health`
- UI:  `http://localhost:5000/fle/`

## Database setup (PostgreSQL)

```bash
createdb fleet_management
psql fleet_management -f capabilities/transport/fle/database/schema.sql
```

Set `DATABASE_URL`:
```bash
export DATABASE_URL=postgresql+asyncpg://user:pass@localhost/fleet_management
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `DEFAULT_TENANT_ID` | `default` | Fallback tenant |
| `DEFAULT_CURRENCY` | `KES` | Default currency code |

## Running tests

```bash
python -m pytest capabilities/transport/fle/tests/ -v
```

## APG Platform deployment

```python
# In APG app factory:
from capabilities.transport.fle import register_capability
register_capability(app, appbuilder)
```

The capability auto-discovers APG adapters (`apg_common_auth`, `apg_common_audl`, etc.) if installed, otherwise falls back to null adapters for standalone operation.
