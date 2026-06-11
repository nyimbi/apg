# Terminal Management System — User Guide

**Capability ID**: `fintech_terminal` | **Domain**: `fintech` | **Version**: `1.2.0`

## Description

Terminal Management System provides a world-class, standalone-deployable implementation of POS terminal lifecycle management, DUKPT/TR-31 key injection, EMV compliance, geo-fenced operation, agency-banking transactions, agent intraday credit, and CBK regulatory reporting for the APG platform.

## Installation

```bash
pip install apg-fintech-terminal
```

## Provides

- `terminal_lifecycle_management`
- `terminal_key_injection_workflow`
- `terminal_parameter_deployment`
- `terminal_certificate_management`
- `terminal_health_monitoring`
- `terminal_transaction_processing`
- `float_and_credit_management`
- `reconciliation_and_reporting`
- `fraud_and_compliance`

## Requires

- `auth` — Token validation and RBAC
- `audl` — Immutable audit log
- `ntfy` — SMS/email/push notifications
- `keym` — HSM key management
- `encr` — Payload encryption helpers

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-terminal/dashboard` | `fintech_terminal:view` | Overview |
| `/fintech-terminal/terminals` | `fintech_terminal:manage` | Terminals |
| `/fintech-terminal/keys` | `fintech_terminal:manage_keys` | Security |
| `/fintech-terminal/parameters` | `fintech_terminal:deploy_parameters` | Configuration |
| `/fintech-terminal/certificates` | `fintech_terminal:manage_certificates` | Security |
| `/fintech-terminal/compliance` | `fintech_terminal:compliance` | Compliance |
| `/fintech-terminal/mobile-money` | `fintech_terminal:mobile_money` | Mobile Money |
| `/fintech-terminal/health` | `fintech_terminal:monitor` | Operations |
| `/fintech-terminal/geo` | `fintech_terminal:manage` | Geo & Security |
| `/fintech-terminal/credit` | `fintech_terminal:float` | Float & Credit |
| `/fintech-terminal/fx` | `fintech_terminal:transactions` | Transactions |
| `/fintech-terminal/reconciliation` | `fintech_terminal:reconcile` | Reconciliation |

## Terminal Lifecycle

### 1. Register

```python
terminal = await svc.register_terminal(
    "T001",
    location={"county": "Nairobi", "address": "Tom Mboya St", "latitude": -1.286, "longitude": 36.817},
    agent_id="AGT-1",
    terminal_type="mpos",
    connectivity="lte",
    serial_number="SN-20260001",
    merchant_id="MID-007",
    model="PAX-A920",
)
```

### 2. Key Injection (DUKPT / TR-31)

Inject an AES-256 key using an HSM-produced BDK reference and KSN.  Raw key material must never be passed through this method.

```python
key = await svc.inject_terminal_key(
    "T001",
    bdk_id="BDK-42",
    ksn="FFFF9876543210E00000",
    key_type="AES256",
    injected_by="HSM-OPS-1",
    expiry_days=365,
)
```

Rotate before the KSN counter exhausts or at annual expiry:

```python
rotation = await svc.rotate_terminal_key(
    "T001",
    new_bdk_id="BDK-43",
    new_ksn="FFFF9876543210E00001",
    key_type="AES256",
    rotated_by="HSM-OPS-1",
)
```

### 3. TLS Certificate

```python
cert = await svc.provision_terminal_certificate(
    "T001", csr_pem=my_csr_pem, issued_by="platform-ca", validity_days=90
)
```

Revoke on tamper detection or key compromise:

```python
await svc.revoke_terminal_certificate(
    "T001", cert["id"], reason="tamper_detected", revoked_by="security_ops"
)
```

### 4. Activate

```python
await svc.activate_terminal("T001", activated_by="field_engineer_7")
```

### 5. OTA Parameter Push

```python
push = await svc.push_terminal_parameters(
    "T001",
    parameters={"bin_table_version": "2026-06", "commission_cash_dep": 0.005},
    pushed_by="ops_admin",
    version="2026-06-11",
    rollback_version="2026-05-01",
)
```

## Transactions

### Cash In / Cash Out

```python
dep = await svc.cash_deposit("T001", customer_id="CUST-1", amount=5000, currency="KES")
wdr = await svc.cash_withdrawal("T001", customer_id="CUST-1", amount=2000, currency="KES")
```

### Mobile Money

```python
await svc.mobile_money_deposit("T001", "CUST-1", 3000, provider="mpesa")
await svc.mobile_money_withdrawal("T001", "CUST-1", 1000, provider="airtel_money")
```

### Cross-Currency (FX)

```python
txn = await svc.foreign_currency_transaction(
    "T001", customer_id="CUST-44",
    amount=100, source_currency="USD",
    target_currency="KES",
    exchange_rate=130.5,
)
```

### Government & Utility Payments

```python
await svc.government_payment("T001", "CUST-1", "NHIF", 500, "MBR-123456")
await svc.nssf_contribution_payment("T001", "CUST-1", employer_ref="EMP-77", amount=200)
await svc.bill_payment_terminal("T001", "CUST-1", biller_code="KPLC", amount=1500)
```

## Float & Agent Credit

### Float Management

```python
await svc.float_management("T001", 50_000, "top_up", authorised_by="supervisor_3")
await svc.float_alert_threshold("T001", min_float=5000, notify_agent=True)
```

### Intraday Credit Facility

When float is insufficient, draw from the agent's approved credit line:

```python
drawdown = await svc.agent_credit_drawdown("AGT-1", "T001", amount=30_000)
```

Repay at EOD settlement:

```python
repayment = await svc.agent_credit_repayment(
    "AGT-1", amount=30_000, reference="EFT-20260610"
)
```

## Geo-Fencing & Relocation

### Geo-fence Check

Called on each heartbeat to detect unauthorised terminal movement:

```python
check = await svc.geo_fence_check("T001", latitude=-1.290, longitude=36.820)
# check["within_fence"] -> True | False
# check["distance_meters"] -> float
```

### Terminal Relocation (dual-approval)

```python
reloc = await svc.relocate_terminal(
    "T001",
    new_location={"latitude": -1.310, "longitude": 36.830, "address": "Moi Ave"},
    requested_by="agent_7",
    approved_by="supervisor_3",
)
```

## Fraud & Velocity

Score every transaction before posting:

```python
verdict = await svc.evaluate_transaction_velocity(
    "T001", customer_id="CUST-99",
    transaction_type="cash_withdrawal", amount=50_000
)
# verdict["recommendation"] -> "allow" | "review" | "deny"
# verdict["fraud_score"]    -> 0–100
# verdict["flags"]          -> list of triggered rules
```

On high-risk events:

```python
await svc.fraud_alert_terminal("T001", "card_skimming_suspected", {"detail": "..."})
```

## Reconciliation & Reporting

### Per-Terminal Daily Reconciliation

```python
recon = await svc.terminal_reconciliation("T001", "2026-06-10")
# recon["status"] -> "balanced" | "variance"
```

### Network-Wide EOD Batch

```python
summary = await svc.batch_reconcile_network("2026-06-10", variance_threshold_pct=0.5)
```

### Agent Commission Report

```python
report = await svc.terminal_commission_report("T001", "2026-Q2")
```

### CBK ABR-01 Regulatory Return

```python
cbk = await svc.cbk_agent_banking_return("2026-Q2", jurisdiction="KE")
```

## Health & Diagnostics

```python
health = await svc.terminal_health_check("T001")
diag  = await svc.pos_diagnostics("T001")
await svc.heartbeat("T001", signal_strength="excellent", battery_pct=82.0)
```

## Key Service Methods

| Method | Description |
|--------|-------------|
| `register_terminal()` | Register a new terminal |
| `activate_terminal()` | Activate after key injection |
| `inject_terminal_key()` | DUKPT/TR-31 key injection |
| `rotate_terminal_key()` | Key rotation with audit chain |
| `provision_terminal_certificate()` | Issue TLS client certificate |
| `revoke_terminal_certificate()` | Revoke certificate |
| `push_terminal_parameters()` | OTA parameter deployment |
| `geo_fence_check()` | GPS boundary enforcement |
| `relocate_terminal()` | Dual-approved relocation |
| `evaluate_transaction_velocity()` | Real-time fraud scoring |
| `cash_deposit()` | Agency cash deposit |
| `cash_withdrawal()` | Agency cash withdrawal |
| `foreign_currency_transaction()` | FX cross-currency transaction |
| `agent_credit_drawdown()` | Intraday float credit draw |
| `agent_credit_repayment()` | EOD credit repayment |
| `batch_reconcile_network()` | Network EOD reconciliation |
| `cbk_agent_banking_return()` | CBK ABR-01 report |
| `terminal_health_check()` | Terminal health poll |
| `fraud_alert_terminal()` | Fraud alert with auto-suspend |
| `suspend_terminal()` | Manual suspension |
| `decommission_terminal()` | Terminal retirement |

_(See `service.py` for complete API and all signatures.)_

## Interoperability

`fintech_terminal` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_terminal;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_TERMINAL_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `FINTECH_TERMINAL_DB_URL` | in-memory | PostgreSQL connection string |
| `FINTECH_TERMINAL_FLOAT_THRESHOLD` | `5000` | Default minimum float alert level (KES) |
| `FINTECH_TERMINAL_GEO_RADIUS_M` | `500` | Default geo-fence radius (metres) |
| `FINTECH_TERMINAL_KEY_EXPIRY_DAYS` | `365` | Default key expiry in days |
| `FINTECH_TERMINAL_CERT_VALIDITY_DAYS` | `90` | Default certificate validity in days |

## Further Reading

- `service.py` — Business logic implementation (all service methods)
- `models.py` — SQLAlchemy and Pydantic data models
- `api.py` — REST API endpoints (Flask-AppBuilder blueprints)
- `views.py` — Flask-AppBuilder views and Pydantic request/response schemas
- `capability_contract.py` — Governance rules and contract definition
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 planned enhancements
