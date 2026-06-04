# Policy Management

## Overview

Policy Management provides a world-class, standalone-deployable implementation of policy management capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Capability ID

`grc_pol`  Version: 1.0.0

## Provides

| Service | Description |
|---------|-------------|
| `policy_lifecycle_management` | Policy Lifecycle Management workflow |
| `policy_acknowledgement_workflow` | Policy Acknowledgement Workflow workflow |
| `policy_exception_workflow` | Policy Exception Workflow workflow |
| `policy_review_workflow` | Policy Review Workflow workflow |
| `policy_publication_workflow` | Policy Publication Workflow workflow |


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
pip install apg-grc-pol
```

## Standalone Usage

```python
from apg_grc_pol import get_capability_contract

# Get capability contract
contract = get_capability_contract(tenant_id="my_org")
print(contract["capability"])  # grc_pol
```

## Running the Standalone Server

```bash
# Standalone with InMemory store
apg-grc-pol --port 8080

# With PostgreSQL persistence
apg-grc-pol --db-url postgresql+asyncpg://user:pass@localhost/pol --port 8080
```

## API Routes

| Name | Path | Permission |
|------|------|------------|
| dashboard | `/grc-pol/dashboard` | `grc_pol:view` |
| policies | `/grc-pol/policies` | `grc_pol:manage_policies` |
| policy_detail | `/grc-pol/policies/:id` | `grc_pol:view` |
| acknowledgements | `/grc-pol/acknowledgements` | `grc_pol:manage_acknowledgements` |
| exceptions | `/grc-pol/exceptions` | `grc_pol:manage_exceptions` |
| reviews | `/grc-pol/reviews` | `grc_pol:review` |
| review_calendar | `/grc-pol/review-calendar` | `grc_pol:view` |
| gap_analysis | `/grc-pol/gap-analysis` | `grc_pol:view` |


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
contract = registry["grc_pol"].contract
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
