# Tax Administration — Installation Guide

© 2025 Datacraft | Author: Nyimbi Odero

---

## Requirements

- Python 3.12+
- PostgreSQL 15+ (for production schema)
- Flask 3.0+
- `uuid6>=0.4`, `pydantic>=2.0`, `sqlalchemy>=2.0`

---

## Standalone Installation

```bash
pip install apg-government-tax
```

Or from source:

```bash
cd capabilities/government/tax
pip install -e .
```

---

## Flask App Integration

```python
from flask import Flask
from apg_government_tax.api import tax_bp
from apg_government_tax.views import views_bp

app = Flask(__name__)
app.register_blueprint(tax_bp)      # REST API at /api/v1/tax/*
app.register_blueprint(views_bp)    # UI views at /tax/*
```

---

## Database Setup

```bash
psql $DATABASE_URL < capabilities/government/tax/database/schema.sql
```

---

## APG Composition Engine

The capability auto-registers via the entry point:

```toml
[project.entry-points."apg.capabilities"]
government_tax = "apg_government_tax:get_capability_contract"
```

Verify registration:

```bash
apg capability list | grep government_tax
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TAX_DEFAULT_TENANT` | `"default"` | Fallback tenant if header missing |
| `TAX_LATE_FILING_RATE` | `0.05` | Override Kenya 5% rate |
| `TAX_LATE_PAYMENT_MONTHLY` | `0.01` | Override Kenya 1%/month rate |

---

## Running Tests

```bash
cd capabilities/government/tax
python -m pytest tests/ -q
# Expected: 292 passed
```
