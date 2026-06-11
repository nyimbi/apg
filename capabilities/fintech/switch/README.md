# Payment Switch

## Overview

Payment Switch is a production-grade ISO 8583 / ISO 20022 payment routing hub for
the APG platform. It handles transaction routing, scheme connectivity, EMV chip
processing, PIN block translation, PAN tokenisation, idempotent authorisation, and
settlement batch management. Deployable standalone or composed with other APG
capabilities.

## Capability ID

`fintech_switch`  Version: 1.2.0

## Provides

| Service | Description |
|---------|-------------|
| `iso8583_message_switching` | ISO 8583 full field parse/build with bitmap |
| `payment_routing_engine` | Multi-rule routing with priority and fallback |
| `channel_key_management` | HSM key injection, rotation, and ZPK derivation |
| `pin_block_translation` | ANSI X9.8 PIN block formats 0/1/3, zone-to-zone translation |
| `mac_generation_verification` | ISO 9797 MAC generation and verification |
| `emv_chip_processing` | ARQC/ARPC cryptogram verification (EMV L3) |
| `pan_tokenisation` | FPE/AES-FF1 tokenisation and de-tokenisation |
| `idempotent_authorisation` | 24-hour idempotency key enforcement |
| `scheme_rate_management` | Interchange rate table CRUD per scheme |
| `circuit_breaker` | Per-network automatic failover with CLOSED/OPEN/HALF_OPEN states |
| `certification_harness` | Parallel scheme certification test runner |
| `settlement_batch` | State-machine driven batch close and clearing file generation |
| `event_bus` | Append-only domain event log with chain hash tamper evidence |


## Requires

| Capability | Purpose |
|------------|---------|
| `auth` | Auth services |
| `audl` | Audit log services |
| `ntfy` | Notification services |
| `keym` | Key management services |
| `encr` | Encryption / HSM services |


## Installation

```bash
pip install apg-fintech-switch
```

## Standalone Usage

```python
from apg_fintech_switch import get_capability_contract
from apg_fintech_switch.service import PaymentSwitchService

# Capability contract
contract = get_capability_contract(tenant_id="my_org")
print(contract["capability"])  # fintech_switch

# Service — in-memory store (development)
svc = PaymentSwitchService(tenant_id="my_org")

# Route a transaction
result = await svc.route_transaction(
    transaction_data={
        "amount": 5000.00, "currency": "KES",
        "transaction_type": "purchase", "channel": "pos",
        "merchant_id": "MID-001", "pan_masked": "4111****1111",
    },
    routing_rules=[
        {"name": "visa_pos", "network": "visa", "priority": 1,
         "conditions": {"currency": "KES", "channel": "pos"}},
    ],
)
```

## Key Service Methods

### Core Routing
| Method | Description |
|--------|-------------|
| `route_transaction` | Route a transaction; assign STAN/RRN; evaluate rules |
| `switch_authorisation` | Authorise from PAN or phone with velocity check |
| `idempotent_authorise` | Authorise with 24-hour idempotency key guarantee |
| `card_not_present_auth` | CNP e-commerce authorisation (CVV/AVS checks) |
| `authenticate_3ds` | Process 3D Secure ECI/CAVV result |

### EMV and Security
| Method | Description |
|--------|-------------|
| `emv_cryptogram_verify` | Verify ARQC and generate ARPC (EMV L3) |
| `pin_verification` | HSM PIN block verification |
| `key_management_hsm` | HSM key inject / generate / rotate |
| `tokenise_pan` | FPE PAN tokenisation (preserves BIN and Luhn) |
| `detokenise_pan` | Retrieve token metadata (no clear-text PAN returned) |

### Routing Management
| Method | Description |
|--------|-------------|
| `routing_table_update` | Atomically replace routing rules (dry_run supported) |
| `scheme_registration` | Register a new payment scheme |
| `scheme_rate_update` | Update interchange rate tables per scheme |
| `acquirer_bin_registration` | Register acquirer BIN ranges |
| `network_interface_register` | Register network endpoint (host/port/protocol) |
| `network_circuit_breaker_status` | Per-network CLOSED/OPEN/HALF_OPEN state |

### Settlement and Clearing
| Method | Description |
|--------|-------------|
| `settlement_batch_close` | Close batch, compute net positions, trigger clearing |
| `clearing_file_generation` | Generate ISO 8583/SWIFT clearing file |
| `settlement_routing` | Route settlement batch to destination bank |
| `reconciliation_switch` | Reconcile switch vs clearing file; flag variance |
| `interchange_fee_calculation` | Calculate interchange fee from rate table |
| `fee_calculation` | All-in fee: interchange + acquirer + issuer |
| `export_settlement_file` | Export clearing file as CSV/JSON/ISO 20022 |
| `daily_settlement_summary` | End-of-day summary across all schemes |

### Fraud and Compliance
| Method | Description |
|--------|-------------|
| `fraud_velocity_check` | Configurable velocity limit check with breach alert |
| `scheme_compliance_check` | Validate transaction against scheme rules |
| `compliance_monitoring` | PCI DSS and scheme compliance dashboard |
| `chargebacks_processing` | File chargeback dispute against an authorization |

### Operations
| Method | Description |
|--------|-------------|
| `switch_health_check` | Component health: routing engine, HSM, networks, queues |
| `load_balancing_status` | TPS distribution across active network interfaces |
| `downtime_failover` | Activate failover; reroute in-flight transactions |
| `network_performance_metrics` | Uptime, latency P99, error rate per network |
| `network_circuit_breaker_status` | Automatic circuit-breaker state per network |

### Analytics and Reporting
| Method | Description |
|--------|-------------|
| `switch_analytics` | Throughput, approval rate, network split, top merchants |
| `switch_analytics_dashboard` | Current-month KPI snapshot |
| `switch_report` | Named report: transaction_summary, approval_rate, etc. |
| `generate_certification_report` | Run parallel scheme certification test suite |
| `transaction_history_switch` | Filtered transaction history query |

### Event Bus
| Method | Description |
|--------|-------------|
| `switch_event_publish` | Publish domain event to append-only log with chain hash |

### ISO Standards
| Method | Description |
|--------|-------------|
| `iso8583_parse` | Parse raw ISO 8583 hex message into structured fields |
| `iso8583_build` | Build ISO 8583 message from MTI and field dict |
| `iso20022_conversion` | Convert ISO 8583 record to ISO 20022 (pacs.008, etc.) |

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
