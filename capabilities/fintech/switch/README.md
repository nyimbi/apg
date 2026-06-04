# Payment Switch

## Overview

Payment Switch provides a world-class, standalone-deployable implementation of payment switch capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Capability ID

`fintech_switch`  Version: 1.1.0

## Provides

| Service | Description |
|---------|-------------|
| `iso8583_message_switching` | Iso8583 Message Switching workflow |
| `payment_routing_engine` | Payment Routing Engine workflow |
| `channel_key_management` | Channel Key Management workflow |
| `pin_block_translation` | Pin Block Translation workflow |
| `mac_generation_verification` | Mac Generation Verification workflow |


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
pip install apg-fintech-switch
```

## Standalone Usage

```python
from apg_fintech_switch import get_capability_contract

# Get capability contract
contract = get_capability_contract(tenant_id="my_org")
print(contract["capability"])  # fintech_switch
```

## Running the Standalone Server

```bash
# Standalone with InMemory store
apg-fintech-switch --port 8080

# With PostgreSQL persistence
apg-fintech-switch --db-url postgresql+asyncpg://user:pass@localhost/switch --port 8080
```

## API Routes

| Name | Path | Permission |
|------|------|------------|
| dashboard | `/fintech-switch/dashboard` | `fintech_switch:view` |
| routing | `/fintech-switch/routing` | `fintech_switch:manage_routing` |
| transactions | `/fintech-switch/transactions` | `fintech_switch:monitor` |
| channels | `/fintech-switch/channels` | `fintech_switch:manage_channels` |
| security | `/fintech-switch/security` | `fintech_switch:manage_keys` |
| mobile_money | `/fintech-switch/mobile-money` | `fintech_switch:mobile_money` |
| settlement | `/fintech-switch/settlement` | `fintech_switch:settle` |
| networks | `/fintech-switch/networks` | `fintech_switch:manage_networks` |


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
contract = registry["fintech_switch"].contract
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
