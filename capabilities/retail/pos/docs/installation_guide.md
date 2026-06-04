# APG Point of Sale — Installation Guide
© 2025 Datacraft | www.datacraft.co.ke

## Requirements

- Python 3.12+
- PostgreSQL 15+ (for production persistence)
- `uv` or `pip` for package management
- `uuid6` package (`pip install uuid6`)

## Standalone Installation

```bash
cd capabilities/retail/pos
pip install -e .
# or with uv:
uv pip install -e .
```

## Run Dev Server

```bash
python -m capabilities.retail.pos --port 8080 --debug
# or via entry point:
apg-retail-pos --port 8080 --debug
```

## Database Setup

```bash
# Create schema
psql $DATABASE_URL -f capabilities/retail/pos/database/schema.sql

# Verify
psql $DATABASE_URL -c "\dt pos.*"
```

For high-traffic tenants, add explicit partitions before first transaction:
```sql
CREATE TABLE pos.pos_transactions_acme
    PARTITION OF pos.pos_transactions
    FOR VALUES IN ('acme-corp');
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string |
| `POS_DEFAULT_TENANT` | `default` | Default tenant ID for standalone mode |
| `POS_VAT_RATE` | `0.16` | Default VAT rate |
| `POS_LOYALTY_EARN_RATE` | `1.0` | Points earned per KES |
| `POS_LOYALTY_REDEEM_RATE` | `0.01` | KES value per point |
| `POS_FLOOR_LIMIT` | `5000.0` | Default terminal floor limit |

## APG Platform Integration

```python
# In APG's app factory:
from capabilities.retail.pos.api import blueprint as pos_api
from capabilities.retail.pos.views import bp as pos_views

app.register_blueprint(pos_api)
app.register_blueprint(pos_views)
```

## Running Tests

```bash
python -m pytest capabilities/retail/pos/tests/ -v
# With coverage:
python -m pytest capabilities/retail/pos/tests/ --cov=capabilities/retail/pos
```
