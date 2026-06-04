# Treasury Management System

## Overview

Treasury Management System provides a world-class, standalone-deployable implementation of treasury management system capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Capability ID

`fintech_treasury`  Version: 1.1.0

## Provides

| Service | Description |
|---------|-------------|
| `cash_position_management` | Cash Position Management workflow |
| `treasury_dealing_workflow` | Treasury Dealing Workflow workflow |
| `counterparty_limit_governance` | Counterparty Limit Governance workflow |
| `settlement_instruction_workflow` | Settlement Instruction Workflow workflow |
| `fx_rate_management` | Fx Rate Management workflow |


## Requires

| Capability | Purpose |
|------------|---------|
| `auth` | Auth services |
| `audl` | Audl services |
| `ntfy` | Ntfy services |
| `keym` | Keym services |
| `fintech_payments` | Fintech Payments services |


## Installation

```bash
pip install apg-fintech-treasury
```

## Standalone Usage

```python
from apg_fintech_treasury import get_capability_contract

# Get capability contract
contract = get_capability_contract(tenant_id="my_org")
print(contract["capability"])  # fintech_treasury
```

## Running the Standalone Server

```bash
# Standalone with InMemory store
apg-fintech-treasury --port 8080

# With PostgreSQL persistence
apg-fintech-treasury --db-url postgresql+asyncpg://user:pass@localhost/treasury --port 8080
```

## API Routes

| Name | Path | Permission |
|------|------|------------|
| dashboard | `/fintech-treasury/dashboard` | `fintech_treasury:view` |
| cash_management | `/fintech-treasury/cash` | `fintech_treasury:manage_cash` |
| dealing | `/fintech-treasury/dealing` | `fintech_treasury:deal` |
| limits | `/fintech-treasury/limits` | `fintech_treasury:manage_limits` |
| settlement | `/fintech-treasury/settlement` | `fintech_treasury:settle` |
| fx | `/fintech-treasury/fx` | `fintech_treasury:manage_fx` |
| liquidity | `/fintech-treasury/liquidity` | `fintech_treasury:manage_liquidity` |
| nostro | `/fintech-treasury/nostro` | `fintech_treasury:reconcile` |


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
contract = registry["fintech_treasury"].contract
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
