# Incident and Case Management

## Overview

Incident and Case Management provides a world-class, standalone-deployable implementation of incident and case management capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Capability ID

`grc_icm`  Version: 1.0.0

## Provides

| Service | Description |
|---------|-------------|
| `incident_lifecycle_management` | Incident Lifecycle Management workflow |
| `case_management_workflow` | Case Management Workflow workflow |
| `incident_evidence_workflow` | Incident Evidence Workflow workflow |
| `regulatory_notification_workflow` | Regulatory Notification Workflow workflow |
| `post_incident_review_workflow` | Post Incident Review Workflow workflow |


## Requires

| Capability | Purpose |
|------------|---------|
| `auth` | Auth services |
| `audl` | Audl services |
| `mten` | Mten services |
| `conf` | Conf services |
| `ntfy` | Ntfy services |


## Installation

```bash
pip install apg-grc-icm
```

## Standalone Usage

```python
from apg_grc_icm import get_capability_contract

# Get capability contract
contract = get_capability_contract(tenant_id="my_org")
print(contract["capability"])  # grc_icm
```

## Running the Standalone Server

```bash
# Standalone with InMemory store
apg-grc-icm --port 8080

# With PostgreSQL persistence
apg-grc-icm --db-url postgresql+asyncpg://user:pass@localhost/icm --port 8080
```

## API Routes

| Name | Path | Permission |
|------|------|------------|
| dashboard | `/grc-icm/dashboard` | `grc_icm:view` |
| incidents | `/grc-icm/incidents` | `grc_icm:manage_incidents` |
| incident_detail | `/grc-icm/incidents/:id` | `grc_icm:view` |
| cases | `/grc-icm/cases` | `grc_icm:manage_cases` |
| case_detail | `/grc-icm/cases/:id` | `grc_icm:view` |
| evidence | `/grc-icm/evidence` | `grc_icm:manage_evidence` |
| notifications | `/grc-icm/notifications` | `grc_icm:view` |
| timeline | `/grc-icm/timeline` | `grc_icm:view` |


## HTTP Endpoints

```
GET  /health           Liveness probe
GET  /contract         Full capability contract JSON
POST /evaluate         Evaluate governance rules
GET  /api/v1/...       Domain-specific REST API
```

## Composability

This capability integrates with the APG platform via the `apg.capabilities` entry-point group. It is auto-discovered by the capability registry when installed.

```python
from capabilities.capability_contract_registry import load_contract_registry
registry = load_contract_registry()
contract = registry["grc_icm"].contract
```

## Development

```bash
# Run tests
pytest tests/ -q

# Build wheel
python -m build --wheel .

# Validate contract
python -c "from capability_contract import get_capability_contract; print('OK')"
```

## License

Proprietary — © 2025 Datacraft  
Author: Nyimbi Odero <nyimbi@gmail.com>
