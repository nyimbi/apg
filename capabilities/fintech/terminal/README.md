# Terminal Management System

## Overview

Terminal Management System provides a world-class, standalone-deployable implementation of terminal management system capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Capability ID

`fintech_terminal`  Version: 1.1.0

## Provides

| Service | Description |
|---------|-------------|
| `terminal_lifecycle_management` | Terminal Lifecycle Management workflow |
| `terminal_key_injection_workflow` | Terminal Key Injection Workflow workflow |
| `terminal_parameter_deployment` | Terminal Parameter Deployment workflow |
| `terminal_certificate_management` | Terminal Certificate Management workflow |
| `terminal_health_monitoring` | Terminal Health Monitoring workflow |


## Requires

| Capability | Purpose |
|------------|---------|
| `auth` | Auth services |
| `audl` | Audl services |
| `ntfy` | Ntfy services |
| `keym` | Keym services |
| `encr` | Encr services |


## Installation

```bash
pip install apg-fintech-terminal
```

## Standalone Usage

```python
from apg_fintech_terminal import get_capability_contract

# Get capability contract
contract = get_capability_contract(tenant_id="my_org")
print(contract["capability"])  # fintech_terminal
```

## Running the Standalone Server

```bash
# Standalone with InMemory store
apg-fintech-terminal --port 8080

# With PostgreSQL persistence
apg-fintech-terminal --db-url postgresql+asyncpg://user:pass@localhost/terminal --port 8080
```

## API Routes

| Name | Path | Permission |
|------|------|------------|
| dashboard | `/fintech-terminal/dashboard` | `fintech_terminal:view` |
| terminals | `/fintech-terminal/terminals` | `fintech_terminal:manage` |
| key_management | `/fintech-terminal/keys` | `fintech_terminal:manage_keys` |
| parameters | `/fintech-terminal/parameters` | `fintech_terminal:deploy_parameters` |
| certificates | `/fintech-terminal/certificates` | `fintech_terminal:manage_certificates` |
| compliance | `/fintech-terminal/compliance` | `fintech_terminal:compliance` |
| mobile_money | `/fintech-terminal/mobile-money` | `fintech_terminal:mobile_money` |
| health | `/fintech-terminal/health` | `fintech_terminal:monitor` |


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
contract = registry["fintech_terminal"].contract
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
