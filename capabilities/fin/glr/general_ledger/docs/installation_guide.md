# General Ledger — Installation Guide

© 2025 Datacraft

---

## System requirements

| Component | Minimum |
|---|---|
| Python | 3.12+ |
| PostgreSQL | 15+ (for production persistence) |
| APG platform | 1.0+ |
| Memory | 512 MB per worker |
| Disk | 10 GB for database |

Python dependencies (from `pyproject.toml`):
- `flask >= 3.0`
- `pydantic >= 2.0`
- `uuid6 >= 2022.10.25`
- `sqlalchemy >= 2.0` (production persistence)
- `psycopg2-binary` (PostgreSQL driver)

---

## Installation within APG platform

### 1. Verify capability is registered

```bash
python -c "from capabilities.fin import glr; print(glr.__version__)"
```

### 2. Apply database schema

```bash
psql -U postgres -d apg_db -f capabilities/fin/glr/general_ledger/database/schema.sql
```

### 3. Register with composition engine

```python
from capabilities.fin.glr.general_ledger.blueprint import register_with_composition_engine
from capabilities.composition.capability_registry import get_registry_service

registry = get_registry_service()
result = register_with_composition_engine(registry)
print(result)  # {"registered": True, "capability_id": "fin.glr.general_ledger", ...}
```

### 4. Mount the Flask blueprint

In your APG application factory:

```python
from capabilities.fin.glr.general_ledger.blueprint import init_subcapability

def create_app(appbuilder):
    init_subcapability(appbuilder)
    # ...
```

Or with a plain Flask app:

```python
from capabilities.fin.glr.general_ledger.api import bp
app.register_blueprint(bp)  # mounts at /api/glr
```

---

## Standalone deployment

For isolated GL deployment without the full APG stack:

### 1. Install dependencies

```bash
cd capabilities/fin/glr/general_ledger
pip install -e ".[server]"
# or
uv pip install -e ".[server]"
```

### 2. Configure environment

```bash
export APG_DATABASE_URL="postgresql://user:pass@localhost:5432/glr_db"
export APG_DEFAULT_TENANT_ID="your-tenant"
export APG_DEFAULT_USER_ID="system"
export FLASK_ENV="production"
```

### 3. Apply schema

```bash
psql $APG_DATABASE_URL -f database/schema.sql
```

### 4. Start server

```bash
# Development
python -m capabilities.fin.glr.general_ledger --port 8080

# or via app.py directly
python capabilities/fin/glr/general_ledger/app.py --port 8080

# Production (gunicorn)
gunicorn "capabilities.fin.glr.general_ledger.app:create_app()" \
    --workers 4 --worker-class gthread --threads 2 \
    --bind 0.0.0.0:8080 --timeout 120
```

### 5. Verify

```bash
curl http://localhost:8080/health
# {"status": "ok", "capability": "glr_general_ledger", ...}

curl -H "X-Tenant-ID: my-tenant" http://localhost:8080/api/glr/accounts
```

---

## Docker deployment

### Using docker-compose

```bash
cd capabilities/fin/glr/general_ledger
docker-compose up -d
```

The included `docker-compose.yml` starts:
- `glr-api`: Flask application (port 8080)
- `postgres`: PostgreSQL 15 (port 5432)

### Environment variables for Docker

```yaml
environment:
  APG_DATABASE_URL: postgresql://glr:glrpass@postgres:5432/glr_db
  APG_DEFAULT_TENANT_ID: default
  FLASK_ENV: production
  GUNICORN_WORKERS: 4
```

---

## Kubernetes deployment

Helm chart and manifests are in the `k8s/` directory:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml     # set APG_DATABASE_URL
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Health check path: `/health`  
Readiness probe: `/api/glr/health`

---

## APG multi-tenant setup

Each tenant has its own isolated data namespace. Configure the default tenant:

```python
svc = GeneralLedgerService(tenant_id="acme", user_id="system")
```

Or via the API by always passing `X-Tenant-ID: acme` in every request.

To initialise a new tenant with a standard chart of accounts:

```python
from capabilities.fin.glr.general_ledger.scripts.seed import seed_chart_of_accounts
seed_chart_of_accounts(svc, tenant_id="acme", framework="IFRS")
```

---

## Integration with other APG modules

| Module | Integration |
|---|---|
| `fin.apy` (Accounts Payable) | AP posts expense journals via `post_journal_v2` |
| `fin.arc` (Accounts Receivable) | AR posts revenue journals and receipts |
| `fin.cbm` (Cash & Bank) | Bank feeds drive bank reconciliation |
| `fin.fam` (Fixed Assets) | Depreciation entries posted monthly |
| `fin.bfc` (Budgeting) | Budget lines read by `budget_vs_actual` |
| `fin.txm` (Tax) | Tax codes applied to journal lines |
| `fin.fco` (Consolidation) | Uses `ifrs_consolidation` for group packs |

---

## Running tests

```bash
# Full suite
uv run pytest -vxs capabilities/fin/glr/general_ledger/tests/

# With coverage
uv run pytest --cov=capabilities/fin/glr/general_ledger \
    --cov-report=html capabilities/fin/glr/general_ledger/tests/

# Type checking
uv run pyright capabilities/fin/glr/general_ledger/
```

---

## Upgrading

1. Back up the database
2. Pull the latest code
3. Apply any new migrations from `database/schema.sql`
4. Restart the service
5. Verify with `/api/glr/health`
