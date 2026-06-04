# Patient Management — Installation Guide

## Requirements

- Python 3.11+
- pip / uv
- PostgreSQL 14+ (optional; in-memory store used by default)
- APG platform (optional; standalone mode works without it)

## Standalone Installation

```bash
pip install apg-healthcare-pmt
# or with uv
uv add apg-healthcare-pmt
```

Start the server:
```bash
apg-healthcare-pmt --port 8080
```

With PostgreSQL:
```bash
apg-healthcare-pmt --port 8080 --db-url postgresql+asyncpg://user:pass@localhost/pmt
```

## APG Platform Installation

Inside the APG platform, install as a capability package:

```bash
pip install apg-healthcare-pmt[full]
```

The entry point `apg.capabilities` group auto-registers the capability:

```toml
[project.entry-points."apg.capabilities"]
healthcare_pmt = "apg_healthcare_pmt:get_capability_contract"
```

Register with the composition engine:
```python
from capabilities.composition import register_capability
from apg_healthcare_pmt import get_capability_contract
register_capability("healthcare_pmt", get_capability_contract)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APG_DATABASE_URL` | None (in-memory) | PostgreSQL connection URL |
| `APG_TENANT_ID` | `default` | Default tenant ID |
| `APG_LOG_LEVEL` | `INFO` | Logging level |

## Database Setup (PostgreSQL)

Run the schema script:
```bash
psql -U postgres -d pmt -f capabilities/healthcare/pmt/database/schema.sql
```

## Development Setup

```bash
git clone https://github.com/datacraft/apg
cd apg/capabilities/healthcare/pmt
uv sync
uv run pytest tests/ -v
```

© 2025 Datacraft
