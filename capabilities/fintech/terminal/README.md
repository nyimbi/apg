# Terminal Management System

## Overview

Terminal Management System provides a world-class, standalone-deployable implementation of POS terminal lifecycle management, DUKPT/TR-31 key injection, EMV compliance, geo-fenced operations, and agency-banking transaction processing for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Capability ID

`fintech_terminal`  Version: 1.2.0

## Provides

| Service | Description |
|---------|-------------|
| `terminal_lifecycle_management` | Register, activate, suspend, relocate, and decommission terminals |
| `terminal_key_injection_workflow` | DUKPT/TR-31 key injection, rotation, and KSN lifecycle |
| `terminal_parameter_deployment` | OTA parameter push with delta compression and rollback |
| `terminal_certificate_management` | TLS client certificate issuance and revocation |
| `terminal_health_monitoring` | Heartbeat, diagnostics, geo-fence, and velocity checks |
| `terminal_transaction_processing` | Deposits, withdrawals, transfers, bill payments, FX |
| `float_and_credit_management` | Float top-up, thresholds, agent intraday credit facility |
| `reconciliation_and_reporting` | Per-terminal and network-wide EOD batch reconciliation |
| `fraud_and_compliance` | Velocity scoring, tamper detection, CBK regulatory returns |


## Requires

| Capability | Purpose |
|------------|---------|
| `auth` | Token validation and RBAC |
| `audl` | Immutable audit event log |
| `ntfy` | SMS/email/push notifications |
| `keym` | HSM key management integration |
| `encr` | Payload encryption helpers |


## Installation

```bash
pip install apg-fintech-terminal
```

## Standalone Usage

```python
from apg_fintech_terminal import get_capability_contract
from apg_fintech_terminal.service import TerminalBankingService

# Capability contract
contract = get_capability_contract(tenant_id="my_org")
print(contract["capability"])  # fintech_terminal

# Service — in-memory (development)
svc = TerminalBankingService(tenant_id="my_org")
terminal = await svc.register_terminal("T001", {"county": "Nairobi"}, "AGT-1", "mpos", "lte")
await svc.inject_terminal_key("T001", bdk_id="BDK-42", ksn="FFFF9876543210E00000", key_type="AES256", injected_by="HSM-OPS-1")
await svc.activate_terminal("T001", activated_by="field_engineer_7")
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
| geo_fence | `/fintech-terminal/geo` | `fintech_terminal:manage` |
| credit | `/fintech-terminal/credit` | `fintech_terminal:float` |
| fx | `/fintech-terminal/fx` | `fintech_terminal:transactions` |
| reconciliation | `/fintech-terminal/reconciliation` | `fintech_terminal:reconcile` |


## HTTP Endpoints

```
GET  /health               Liveness probe
GET  /contract             Full capability contract JSON
POST /evaluate             Evaluate governance rules
GET  /api/v1/...           Domain-specific REST API
POST /api/v1/keys/inject   Key injection endpoint
POST /api/v1/keys/rotate   Key rotation endpoint
POST /api/v1/certs/issue   Certificate provisioning
POST /api/v1/certs/revoke  Certificate revocation
POST /api/v1/geo/check     Geo-fence validation
POST /api/v1/recon/batch   Network-wide EOD reconciliation
POST /api/v1/credit/draw   Agent credit drawdown
POST /api/v1/credit/repay  Agent credit repayment
POST /api/v1/fx            Cross-currency transaction
```

## New Features (v1.2.0)

### DUKPT / TR-31 Key Injection

```python
# Inject a new AES-256 key
key = await svc.inject_terminal_key(
    "T001", bdk_id="BDK-42",
    ksn="FFFF9876543210E00000",
    key_type="AES256",
    injected_by="HSM-OPS-1",
    expiry_days=365,
)

# Rotate when KSN counter nears exhaustion
rotation = await svc.rotate_terminal_key(
    "T001", new_bdk_id="BDK-43",
    new_ksn="FFFF9876543210E00001",
    key_type="AES256",
    rotated_by="HSM-OPS-1",
)
```

### TLS Certificate Lifecycle

```python
cert = await svc.provision_terminal_certificate(
    "T001", csr_pem=my_csr_pem, issued_by="platform-ca", validity_days=90
)
# Revoke on tamper
await svc.revoke_terminal_certificate("T001", cert["id"], reason="tamper_detected", revoked_by="security_ops")
```

### OTA Parameter Push

```python
push = await svc.push_terminal_parameters(
    "T001",
    parameters={"bin_table_version": "2026-06", "commission_cash_dep": 0.005},
    pushed_by="ops_admin",
    version="2026-06-11",
    rollback_version="2026-05-01",
)
```

### Geo-Fence Enforcement

```python
# Verify terminal hasn't moved more than 500 m
check = await svc.geo_fence_check("T001", latitude=-1.286, longitude=36.817)
# Relocate with dual approval
await svc.relocate_terminal(
    "T001",
    new_location={"latitude": -1.290, "longitude": 36.820, "address": "Tom Mboya St"},
    requested_by="agent_7",
    approved_by="supervisor_3",
)
```

### Transaction Velocity & Fraud Scoring

```python
verdict = await svc.evaluate_transaction_velocity(
    "T001", customer_id="CUST-99", transaction_type="cash_withdrawal", amount=50_000
)
# verdict["recommendation"] -> "allow" | "review" | "deny"
# verdict["fraud_score"]    -> 0–100
```

### EOD Network Batch Reconciliation

```python
summary = await svc.batch_reconcile_network("2026-06-10", variance_threshold_pct=0.5)
# Returns aggregate credits, debits, flagged terminals with variances
```

### Agent Intraday Credit Facility

```python
drawdown = await svc.agent_credit_drawdown("AGT-1", "T001", amount=30_000)
repayment = await svc.agent_credit_repayment("AGT-1", amount=30_000, reference="EFT-20260610")
```

### Cross-Currency (FX) Transactions

```python
txn = await svc.foreign_currency_transaction(
    "T001", customer_id="CUST-44",
    amount=100, source_currency="USD",
    target_currency="KES", exchange_rate=130.5,
)
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
